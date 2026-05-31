import csv
from utils.database import get_connection


def export_csv():

    conn = get_connection()

    rows = conn.execute(
        "SELECT * FROM predictions"
    ).fetchall()

    conn.close()

    filename = "export.csv"

    with open(filename,"w",newline="") as f:

        writer = csv.writer(f)

        writer.writerow(rows[0].keys())

        for row in rows:
            writer.writerow(row)

    return filename
