import fitz  # PyMuPDF
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR"
from PIL import Image
import io

def extract_text_from_pdf(file) -> str:
    text = ""
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")

    for page in doc:
        # Try normal text extraction first
        page_text = page.get_text()
        if page_text and page_text.strip():
            text += page_text + "\n"
        else:
            # Use OCR if the page is scanned/image
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text + "\n"

    return text.strip()
