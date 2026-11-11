"""Parse uploaded PDF bytes in a disposable child process, with parent deadline."""
import base64
from io import BytesIO
import json
import sys

from pypdf import PdfReader


def main():
    try:
        payload = json.loads(sys.stdin.read(30*1024*1024))
        content = base64.b64decode(payload["content"], validate=True)
        if len(content) > 20*1024*1024 or not content.startswith(b"%PDF-"):
            raise ValueError("Invalid or oversized PDF")
        reader = PdfReader(BytesIO(content), strict=True)
        if reader.is_encrypted or len(reader.pages) > 100:
            raise ValueError("Encrypted PDFs and PDFs over 100 pages are not supported")
        pages, total = [], 0
        for number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").replace("\x00", "")
            total += len(text)
            if total > 500000:
                raise ValueError("Extracted PDF text exceeds the document budget")
            if text.strip():
                pages.append({"page": number, "text": text})
        if not pages:
            raise ValueError("No extractable text; scanned PDFs need a separately configured OCR pipeline")
        print(json.dumps({"pages": pages}, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"error": str(error)[:300]}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
