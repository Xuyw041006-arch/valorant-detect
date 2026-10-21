"""Download EasyOCR's standard English inference models once (no training)."""

import easyocr

if __name__ == "__main__":
    easyocr.Reader(["en"], gpu=False, download_enabled=True, verbose=True)
    print("English OCR models are ready.")
