"""Prepare and test a fixed, deduplicated sample from the user-selected invoice sources."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageOps

from qa_archive_invoices import ROOT, benchmark, save_json


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    downloads = Path.home() / "Downloads"
    selected, seen, duplicates = [], set(), []

    def add(data, filename, origin, annotation=None):
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            duplicates.append({"filename": filename, "origin": origin, "sha256": digest})
            return
        seen.add(digest)
        destination = output / filename
        destination.write_bytes(data)
        selected.append({"entry": filename, "local_path": str(destination), "sha256": digest, "origin": origin})
        if annotation is not None:
            # Ground truth is saved separately and is never supplied to OCR, retrieval, or generation.
            save_json(output / (Path(filename).stem + ".annotation.json"), annotation)

    with zipfile.ZipFile(downloads / "dataset_reduit-20260911T095608Z-1-001.zip") as archive:
        for name in ("facture_00008.png", "facture_00158.png", "facture_00540.png"):
            entry = "dataset_reduit/images/val/" + name
            add(archive.read(entry), name, {"archive": archive.filename, "entry": entry, "split": "val"})
    with zipfile.ZipFile(downloads / "FATURA.zip") as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("invoices_dataset_final/strat1_test.csv").decode("utf-8-sig"))))
        for template in (1, 26):
            row = next(row for row in rows if row["img_path"].startswith(f"Template{template}_"))
            entry = "invoices_dataset_final/images/" + row["img_path"]
            truth = json.loads(archive.read("invoices_dataset_final/Annotations/Original_Format/" + row["annot_path"]))
            add(archive.read(entry), row["img_path"], {"archive": archive.filename, "entry": entry, "split": "strat1_test"}, truth)
    for name in ("facture_00079.png", "facture_00040.png", "facture_00791.png", "facture_00158 (1).png"):
        add((downloads / name).read_bytes(), name, {"file": str(downloads / name)})
    save_json(output / "manifest.json", {"selected": selected, "duplicates_excluded": duplicates,
        "selection": "Fixed dataset validation names, first test example from FATURA templates 1 and 26, and named Downloads invoices"})
    sheet = Image.new("RGB", (1400, 640 * 4), "#e9eef5")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(selected):
        x, y = (index % 2) * 700, (index // 2) * 640
        draw.text((x + 12, y + 8), item["entry"], fill="black")
        with Image.open(item["local_path"]) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((675, 600))
            sheet.paste(preview, (x + (700 - preview.width) // 2, y + 30))
    sheet.save(output / "contact-sheet.jpg", quality=94)
    print(json.dumps({"selected": [item["entry"] for item in selected], "duplicates_excluded": len(duplicates)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "qa-results" / "groq-multisource-20260912")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if args.run:
        benchmark(args.output, [], paddle=True, reuse_ocr=False, provider="groq")
    else:
        prepare(args.output)
