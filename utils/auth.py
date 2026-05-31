from werkzeug.security import generate_password_hash, check_password_hash
from utils.database import get_connection


def create_user(username, password):

    conn = get_connection()

    # hash password
    hashed = generate_password_hash(password)

    try:

        conn.execute(
            "INSERT INTO users(username,password) VALUES (?,?)",
            (username, hashed)
        )

        conn.commit()

        conn.close()

        return True

    except:

        conn.close()

        return False



def authenticate(username, password):

    conn = get_connection()

    user = conn.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    ).fetchone()

    conn.close()

    if user and check_password_hash(user["password"], password):

        return True

    return False
