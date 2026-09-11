
from pathlib import Path
import subprocess
import sys
import os

def extract_pdf(pdf_path: str, output_dir: str = "./output1/pdf_extract"):
    env = os.environ.copy()
    os.environ["MINERU_PDF_RENDER_THREADS"] = "1"
    os.environ["MINERU_API_MAX_CONCURRENT_REQUESTS"] = "1"
    os.environ["MINERU_DEVICE_MODE"] = "cpu"
    pdf = Path(pdf_path)

    if not pdf.exists():
        print(f"Error: PDF not found: {pdf}")
        sys.exit(1)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Extracting: {pdf}")

    # Run MinerU
    command = [
        "mineru",
        "-p", str(pdf),
        "-o", str(output),
        "-b", "pipeline"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env
    )

    if result.returncode != 0:
        print("MinerU failed:")
        print(result.stderr)
        sys.exit(result.returncode)

    print("Extraction completed!")

    # Find generated Markdown files
    markdown_files = list(output.rglob("*.md"))

    if not markdown_files:
        print("No Markdown file was generated.")
        return

    for md_file in markdown_files:
        print(f"\nExtracted content: {md_file}")

        content = md_file.read_text(encoding="utf-8")

        print("\n--- CONTENT ---\n")
        print(content)

        return content


if __name__ == "__main__":
    extract_pdf("BAC-CE-CMR-2017.pdf")












