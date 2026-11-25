"""List the PDFs you will give to Forge, extract their text, and start the licence log.

Why: the judges score dataset quality and the release must be reproducible; you must also have the right to
republish what you derive from each document. This script makes data/build/docs/sources.csv, which you complete by
hand (publisher, URL, licence, OK to publish?). Only documents marked ok_to_publish=yes go to Forge.

Run:  python -m src.pdf_inventory --pdf-dir data/raw/fiches --out data/build/docs
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main(pdf_dir: str, out: str):
    from pypdf import PdfReader  # pip install pypdf

    out_dir = Path(out)
    (out_dir / "text").mkdir(parents=True, exist_ok=True)
    rows = []
    for pdf in sorted(Path(pdf_dir).glob("**/*.pdf")):
        try:
            reader = PdfReader(str(pdf))
            pages = [p.extract_text() or "" for p in reader.pages]
        except Exception as e:  # damaged or protected file
            print(f"SKIP {pdf.name}: {e}")
            continue
        text = "\n\n".join(f"[page {i + 1}]\n{t}" for i, t in enumerate(pages))
        (out_dir / "text" / (pdf.stem + ".txt")).write_text(text, encoding="utf-8")
        chars = sum(len(t) for t in pages)
        rows.append({"file": pdf.name, "pages": len(pages), "chars": chars,
                     "scanned_needs_ocr": "yes" if chars < 200 * max(1, len(pages)) else "no",
                     "title": "", "publisher": "", "year": "", "url": "", "licence": "", "ok_to_publish": "",
                     "crops_topics": "", "language": ""})
    with open(out_dir / "sources.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} PDFs -> {out_dir / 'sources.csv'} (complete the empty columns by hand)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="data/raw/fiches")
    ap.add_argument("--out", default="data/build/docs")
    a = ap.parse_args()
    main(a.pdf_dir, a.out)
