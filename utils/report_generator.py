from reportlab.pdfgen import canvas
from config import REPORT_FOLDER
import os


def generate_pdf(id, result, cancer_prob, normal_prob, image, heatmap):

    path = os.path.join(REPORT_FOLDER, f"report_{id}.pdf")

    c = canvas.Canvas(path)

    c.drawString(100,800,"Oral Cancer Detection Report")

    c.drawString(100,750,f"Result: {result}")
    c.drawString(100,730,f"Cancer Probability: {cancer_prob}%")
    c.drawString(100,710,f"Normal Probability: {normal_prob}%")

    c.drawImage(image,100,500,200,150)
    c.drawImage(heatmap,320,500,200,150)

    c.save()

    return path
