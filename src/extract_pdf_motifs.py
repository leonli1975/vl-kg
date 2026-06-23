"""Extract figures from PDF using pre-existing OCR text + image processing.

Strategy (no runtime OCR needed):
  1. Parse existing OCR text → get figure captions with page numbers
  2. Render PDF pages as images  
  3. Detect figure regions using image processing (entropy + edge detection)
  4. Match figures to captions by page and position
  5. Save figures + metadata

Output:
  data/extracted/<book_name>/
    ├── images/
    │   ├── 图1-1.jpg
    │   └── ...
    └── metadata.json
"""

import json, os, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from PIL import Image, ImageFilter
import pypdfium2 as pdfium

RENDER_DPI = 200
FIGURE_CAPTION_PATTERN = re.compile(r'图\s*(\d+)\s*[-−–—]\s*(\d+)')
MIN_FIGURE_SIZE = (120, 120)


def parse_captions_from_ocr(ocr_path: str) -> Dict[str, dict]:
    """Parse figure captions from pre-existing OCR text file.

    Returns: {fig_id: {caption, page, is_standalone}}
    """
    with open(ocr_path) as f:
        text = f.read()

    pages = re.split(r'=== 第 (\d+) 页 ===', text)
    figures_raw = defaultdict(list)
    current_page = 0

    for i, chunk in enumerate(pages):
        if i % 2 == 1:
            current_page = int(chunk)
        else:
            for m in FIGURE_CAPTION_PATTERN.finditer(chunk):
                fig_id = f"图{m.group(1)}-{m.group(2)}"
                # Get context: text after the match (caption description)
                after = chunk[m.end():m.end() + 100].strip()
                before = chunk[max(0, m.start() - 30):m.start()].strip()

                # Is it a standalone caption? Standalone captions typically:
                # - Have "图X-X" near the start of a line
                # - Are followed by descriptive text (not just punctuation)
                is_standalone = (
                    len(before) < 15 or '\n' in before[-5:] or
                    bool(re.match(r'^[^\w]{0,5}$', before[-10:]))
                )

                # Extract caption text
                if is_standalone:
                    caption_text = (f"图{m.group(1)}-{m.group(2)}" + after).strip()
                    # Truncate at newline or next figure
                    caption_text = re.split(r'[\n]', caption_text)[0]
                    caption_text = re.split(r'图\d+[-−–—]\d+', caption_text)[0]
                else:
                    # Inline reference: extract surrounding sentence
                    start = max(0, m.start() - 40)
                    end = min(len(chunk), m.end() + 60)
                    caption_text = chunk[start:end].replace('\n', ' ').strip()

                caption_text = caption_text[:200]  # limit length

                figures_raw[fig_id].append({
                    "page": current_page,
                    "caption": caption_text,
                    "is_standalone": is_standalone,
                })

    # Deduplicate: prefer standalone captions
    captions = {}
    for fig_id, entries in figures_raw.items():
        standalone = [e for e in entries if e["is_standalone"]]
        if standalone:
            best = max(standalone, key=lambda e: len(e["caption"]))
        else:
            best = entries[0]
        captions[fig_id] = best

    return captions


def detect_content_regions(pil_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect non-white content regions (likely figures) in a page image.

    Uses horizontal projection + edge detection to find distinct
    content blocks.

    Returns: list of (x0, y0, x1, y1) bounding boxes
    """
    h, w = pil_image.shape[:2]
    gray = pil_image if len(pil_image.shape) == 2 else np.mean(pil_image, axis=2)

    # Horizontal projection: count non-white pixels per row
    threshold = 230
    non_white_rows = np.mean(gray < threshold, axis=1)

    # Find continuous blocks of rows with significant content
    content_threshold = 0.05  # at least 5% non-white pixels
    in_block = False
    blocks = []
    block_start = 0

    for y in range(h):
        is_content = non_white_rows[y] > content_threshold
        if is_content and not in_block:
            block_start = y
            in_block = True
        elif not is_content and in_block:
            if y - block_start > 30:  # minimum block height
                # Find left/right bounds
                row_slice = gray[block_start:y, :]
                col_content = np.mean(row_slice < threshold, axis=0) > 0.02
                content_cols = np.where(col_content)[0]
                if len(content_cols) > 20:
                    x0 = max(0, content_cols[0] - 10)
                    x1 = min(w, content_cols[-1] + 10)
                    blocks.append((x0, block_start, x1, y))
            in_block = False

    # Filter: keep blocks with reasonable aspect ratio (not full-width text lines)
    fig_blocks = []
    for x0, y0, x1, y1 in blocks:
        bw = x1 - x0
        bh = y1 - y0
        if bw < MIN_FIGURE_SIZE[0] or bh < MIN_FIGURE_SIZE[1]:
            continue
        # Skip blocks that span almost full width (likely text paragraphs)
        if bw > w * 0.85 and bh < 100:
            continue
        fig_blocks.append((x0, y0, x1, y1))

    return fig_blocks


def extract_page_figures(pdf, page_num: int, captions_on_page: List[dict],
                          images_dir: Path, metadata: dict) -> int:
    """Extract figures from a single page, matching to captions."""
    page = pdf[page_num]
    scale = RENDER_DPI / 72
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    arr = np.array(pil_image)

    # Detect content regions
    regions = detect_content_regions(arr)

    if not regions:
        return 0

    # Sort regions by Y position
    regions.sort(key=lambda r: r[1])

    # Also sort captions by their order (they usually appear top to bottom)
    # Match each region to a caption (same order assumption)
    n = min(len(regions), len(captions_on_page))
    extracted = 0

    for i in range(n):
        x0, y0, x1, y1 = regions[i]
        caption_info = captions_on_page[i]

        # Add small padding
        px = 5
        x0 = max(0, x0 - px)
        y0 = max(0, y0 - px)
        x1 = min(pil_image.width, x1 + px)
        y1 = min(pil_image.height, y1 + px)

        fig_img = pil_image.crop((x0, y0, x1, y1))
        fig_id = caption_info["fig_id"]
        safe_id = fig_id.replace("图", "fig_")
        img_path = images_dir / f"{safe_id}.jpg"
        fig_img.save(str(img_path), quality=90)

        entry = {
            "fig_id": fig_id,
            "page": page_num + 1,
            "caption": caption_info["caption"],
            "bbox_page": [int(x0), int(y0), int(x1), int(y1)],
            "image_path": str(img_path.relative_to(images_dir.parent)),
        }
        metadata[fig_id] = entry
        print(f"    ✓ {fig_id} (第{page_num+1}页) → {safe_id}.jpg "
              f"({x1-x0}×{y1-y0}px)")
        extracted += 1

    return extracted


def extract_pdf(pdf_path: str, ocr_path: str = None, output_dir: str = None):
    """Main extraction pipeline."""
    pdf_path = Path(pdf_path)
    book_name = pdf_path.stem

    if ocr_path is None:
        # Try to find matching OCR file
        ocr_path = pdf_path.parent / f"{book_name}_output.txt"
        if not ocr_path.exists():
            # Try alternate naming
            for candidate in pdf_path.parent.glob("*output*.txt"):
                ocr_path = candidate
                break

    if not ocr_path or not Path(ocr_path).exists():
        print(f"错误: 找不到OCR文本文件: {ocr_path}")
        print("请确保 data/ 目录下有对应的 *_output.txt 文件")
        return 0

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "extracted" / book_name

    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse captions from OCR text
    print(f"\n[1] 解析OCR文本: {Path(ocr_path).name}")
    captions = parse_captions_from_ocr(str(ocr_path))
    print(f"    找到 {len(captions)} 个独有图注")

    # Group by page
    by_page = defaultdict(list)
    for fig_id, info in captions.items():
        by_page[info["page"]].append({"fig_id": fig_id, **info})

    # Step 2: Extract figures page by page
    print(f"\n[2] 提取图像...")
    pdf = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(pdf)
    metadata = {}
    total_extracted = 0

    for page_num in range(total_pages):
        page_1idx = page_num + 1
        if page_1idx in by_page:
            page_captions = by_page[page_1idx]
            n = extract_page_figures(
                pdf, page_num, page_captions, images_dir, metadata
            )
            total_extracted += n

    # Step 3: Save metadata
    meta_path = Path(output_dir) / "metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[3] 完成")
    print(f"    PDF页数: {total_pages}")
    print(f"    有图页面: {len(by_page)}")
    print(f"    提取图像: {total_extracted}")
    print(f"    输出目录: {output_dir}")
    print(f"    元数据:   {meta_path}")

    return total_extracted


def main():
    import argparse
    parser = argparse.ArgumentParser(description="从PDF中提取图像和图注")
    parser.add_argument("pdf", nargs="?", help="PDF文件路径")
    parser.add_argument("--ocr", help="OCR文本文件路径 (默认自动查找)")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--list", action="store_true", help="列出可用PDF")
    args = parser.parse_args()

    if args.list:
        data_dir = Path(__file__).parent.parent / "data"
        for pdf in sorted(data_dir.glob("*.pdf")):
            ocr_files = list(data_dir.glob(f"{pdf.stem}*output*"))
            ocr_info = f" (OCR: {ocr_files[0].name})" if ocr_files else " (无OCR)"
            print(f"  {pdf.name}{ocr_info}")
        return

    if not args.pdf:
        # Default: process both books
        data_dir = Path(__file__).parent.parent / "data"
        pdfs = [
            (data_dir / "蜀锦.pdf", data_dir / "sj_output.txt"),
        ]
        # Check second book
        zg = data_dir / "中国成都蜀锦   11798555.pdf"
        zg_ocr = data_dir / "zgcdsj_output.txt"
        if zg.exists():
            pdfs.append((zg, zg_ocr))

        total = 0
        for pdf_path, ocr_path in pdfs:
            if pdf_path.exists():
                print("=" * 64)
                print(f"处理: {pdf_path.name}")
                print("=" * 64)
                total += extract_pdf(str(pdf_path), str(ocr_path) if ocr_path.exists() else None)
        print(f"\n总计提取: {total} 张图像")
        return

    extract_pdf(args.pdf, args.ocr, args.output)


if __name__ == "__main__":
    main()
