import os
import cv2
import numpy as np
import io
from flask import Flask, request, render_template, send_file
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

app = Flask(__name__)


def process_fingerprint(image_file):
    # Convert file to OpenCV format
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Apply processing (thresholding)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    img_thresh = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Convert processed image to bytes
    _, img_encoded = cv2.imencode(".png", img_thresh, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return io.BytesIO(img_encoded.tobytes())


def create_pdf(data):
    pdf_bytes = io.BytesIO()
    c = canvas.Canvas(pdf_bytes, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)

    page_width, page_height = letter
    margin_top = 50  # Top margin
    y_position = page_height - margin_top  # Start position

    for person in data:
        name = person["name"]
        c.drawString(50, y_position, name)  # Draw name

        # Calculate image placement dynamically
        max_fingerprints = len(person["fingerprints"])
        max_width = min(
            110, (page_width - 250) // max_fingerprints
        )  # Adjust width based on count
        max_height = 130  # Maintain height

        x_pos = 200  # Start after name
        for img in person["fingerprints"]:
            c.drawImage(
                ImageReader(img),
                x_pos,
                y_position - max_height + 10,
                width=max_width,
                height=max_height,
                preserveAspectRatio=True,
                mask="auto",
            )
            x_pos += max_width + 10  # Adjust spacing dynamically

        y_position -= max_height + 30  # Move to next row

        # Start new page if needed
        if y_position < 100:
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            y_position = page_height - margin_top

    c.save()
    pdf_bytes.seek(0)
    return pdf_bytes


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        names = request.form.getlist("names[]")
        processed_data = []

        for i, name in enumerate(names):
            files = request.files.getlist(f"fingerprints_{i}[]")

            if not (1 <= len(files) <= 4):
                return (
                    f"Each person must upload 1 to 4 fingerprints. Error for {name}",
                    400,
                )

            processed_data.append(
                {
                    "name": name,
                    "fingerprints": [process_fingerprint(file) for file in files],
                }
            )

        pdf_bytes = create_pdf(processed_data)
        return send_file(
            pdf_bytes,
            as_attachment=True,
            download_name="fingerprints_list.pdf",
            mimetype="application/pdf",
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
    )
