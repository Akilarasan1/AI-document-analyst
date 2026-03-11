import pytesseract
from PIL import Image
from langchain.schema import Document
import os
import shutil


def find_tesseract_executable():
    """
    Check if Tesseract is installed and set its path.
    Returns True if found, otherwise raises an exception.
    """
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        return True

    tesseract_env_path = os.getenv("TESSERACT_PATH")
    if tesseract_env_path and os.path.exists(tesseract_env_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_env_path
        return True

    common_paths = [
        os.path.expanduser("~\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"),  # Typical Windows path
        "/usr/local/lib/python3.10/dist-packages/pytesseract/tesseract.exe",  # Google Colab path
        "/usr/bin/tesseract",  # Common Linux path
    ]
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True

    raise FileNotFoundError("Tesseract executable not found. Please install Tesseract OCR or set TESSERACT_PATH environment variable.")


def load_image_document(image_path):

    img = Image.open(image_path)
    if find_tesseract_executable():
        text = pytesseract.image_to_string(img)

    doc = Document(
        page_content=text,
        metadata={"source": image_path}
    )

    return [doc]