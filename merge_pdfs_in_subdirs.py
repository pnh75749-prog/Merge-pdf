from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def merge_pdf_files(pdf_files, output_path: Path) -> None:
    writer = PdfWriter()

    for pdf_file in sorted(pdf_files, key=lambda item: item.name.lower()):
        reader = PdfReader(str(pdf_file))
        for page in reader.pages:
            writer.add_page(page)

    with output_path.open("wb") as file:
        writer.write(file)


def merge_subdirectory_pdfs(main_dir: Path) -> int:
    merged_count = 0

    for folder in sorted(main_dir.iterdir(), key=lambda item: item.name.lower()):
        if not folder.is_dir():
            continue

        pdf_files = sorted(
            list(folder.glob("*.pdf")) + list(folder.glob("*.PDF")),
            key=lambda item: item.name.lower(),
        )

        if not pdf_files:
            continue

        merged_file = main_dir / f"{folder.name}.pdf"

        if merged_file in pdf_files:
            pdf_files = [item for item in pdf_files if item != merged_file]

        if not pdf_files:
            continue

        merge_pdf_files(pdf_files, merged_file)
        merged_count += 1
        print(f"Merged {len(pdf_files)} files in '{folder.name}' -> '{merged_file.name}'")

    return merged_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge all PDF files in each subdirectory of a chosen main directory. "
        "The merged file name will match the subdirectory name."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Main directory that contains PDF folders (for example: D:/August - 2026)",
    )
    args = parser.parse_args()

    main_dir = args.directory.resolve()

    if not main_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {main_dir}")

    if not main_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {main_dir}")

    count = merge_subdirectory_pdfs(main_dir)
    print(f"\nDone. Merged {count} subdirectories.")


if __name__ == "__main__":
    main()
