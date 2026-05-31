import tensorflow as tf
import numpy as np
import cv2

LAST_CONV_LAYER = "conv2d_3"


def make_gradcam_heatmap(img_array, model):
    """
    Fixed GradCAM: uses tf.GradientTape directly on the full model
    by extracting intermediate output via a submodel built from
    the actual model's layers (not re-applying them on new Input).
    """

    # Build a model that outputs [last_conv_output, final_prediction]
    # by using the model's existing layer graph
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(LAST_CONV_LAYER).output,
            model.output
        ]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)

    # Avoid division by zero if heatmap is all zeros
    heatmap = tf.cond(
        max_val > 0,
        lambda: heatmap / max_val,
        lambda: heatmap
    )

    return heatmap.numpy()


def overlay_heatmap(original_path, heatmap, output_path):

    img = cv2.imread(original_path)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    cv2.imwrite(output_path, superimposed_img)

    return output_path