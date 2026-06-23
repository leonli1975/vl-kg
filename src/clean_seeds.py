"""Clean seed images using VL motif decomposition, then re-seed.

For each motif directory, VL verifies whether the target motif is dominant.
Only "clean" images (target = top-1, confidence > threshold) are kept.

Pipeline:
  1. VL scan: for each image, check if target motif is top-1
  2. Report: per-motif clean/noisy counts
  3. Re-seed: use only clean images as visual centroids
  4. Re-evaluate: run A/B comparison with cleaned seeds

Usage:
  python -m src.clean_seeds                    # clean + re-seed + evaluate
  python -m src.clean_seeds --dry-run          # only report, don't re-seed
"""

import json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np

from .vl_recognizer import VLMotifRecognizer
from .enhanced_kg import EnhancedThreeLayerKG
from .three_layer_kg import ATOMIC_MOTIFS

MOTIFS_DIR = Path(__file__).parent.parent / "data" / "images" / "motifs"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval_results"

DIR_TO_NAME = {
    "凤凰纹": "凤凰纹", "龙纹": "龙纹", "蝴蝶纹": "蝴蝶纹",
    "麒麟纹": "麒麟纹", "仙鹤纹": "仙鹤纹", "鸳鸯纹": "鸳鸯纹",
    "孔雀纹": "孔雀纹", "鹿纹": "鹿纹", "狮纹": "狮纹",
    "卷草纹": "卷草纹", "牡丹纹": "牡丹纹", "莲花纹": "莲花纹",
    "宝相花纹": "宝相花纹", "梅花纹": "梅花纹",
    "云纹": "云纹", "万字纹": "万字纹", "回纹": "回纹",
    "龟背纹": "龟背纹", "联珠纹": "联珠纹",
    "龟背纹": "龟背纹", "联珠纹": "联珠纹",
    "文字纹": "文字纹",
}

CONFIDENCE_THRESHOLD = 0.6


def scan_and_clean(recognizer: VLMotifRecognizer) -> dict:
    """Scan all motif directories, classify images as clean or noisy.

    Returns per-motif: {motif_name: {clean: [paths], noisy: [paths], stats: {}}}
    """
    results = {}

    for dir_path in sorted(MOTIFS_DIR.iterdir()):
        if not dir_path.is_dir():
            continue
        dir_name = dir_path.name
        gt_name = DIR_TO_NAME.get(dir_name)
        if not gt_name:
            continue

        images = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            images.extend(sorted(dir_path.glob(ext)))

        if not images:
            continue

        clean = []
        noisy = []
        stats = []

        for img in images:
            try:
                vl_result = recognizer.decompose_motifs(str(img))
                motifs = vl_result.get("motifs", [])
                names = [m.get("name", "") for m in motifs]
                confs = {m.get("name", ""): m.get("confidence", 0) for m in motifs}

                top1_name = names[0] if names else None
                top1_conf = confs.get(top1_name, 0) if top1_name else 0
                gt_in_list = gt_name in names
                gt_rank = names.index(gt_name) + 1 if gt_in_list else None
                gt_conf = confs.get(gt_name, 0)

                is_clean = (top1_name == gt_name and top1_conf >= CONFIDENCE_THRESHOLD)

                item = {
                    "image": img.name,
                    "top1": top1_name,
                    "top1_conf": top1_conf,
                    "gt_rank": gt_rank,
                    "gt_conf": gt_conf,
                    "all_motifs": names,
                    "is_clean": is_clean,
                }

                if is_clean:
                    clean.append(str(img))
                else:
                    noisy.append(str(img))

                stats.append(item)

                mark = "✓" if is_clean else "✗"
                extra = f" (rank={gt_rank}, conf={gt_conf:.2f})" if gt_in_list else " (not found)"
                print(f"  {mark} {gt_name}/{img.name}: top1={top1_name}{extra}")

            except Exception as e:
                print(f"  ! {gt_name}/{img.name}: ERROR - {e}")
                noisy.append(str(img))

        n_clean = len(clean)
        n_total = len(images)
        results[gt_name] = {
            "clean": clean,
            "noisy": noisy,
            "n_clean": n_clean,
            "n_total": n_total,
            "pct_clean": n_clean / n_total if n_total else 0,
            "details": stats,
        }

        print(f"  → {gt_name}: {n_clean}/{n_total} clean ({n_clean/n_total:.0%})")

    return results


def reseed_with_clean(results: dict):
    """Re-seed EnhancedThreeLayerKG using only clean images."""
    print(f"\n{'='*64}")
    print("用精化后的种子重新播种")
    print(f"{'='*64}")

    kg = EnhancedThreeLayerKG()
    from .seed_motifs import DIR_TO_MOTIF as SEED_DIR_TO_MOTIF
    from .evaluate_vl import _name_to_key

    total_before = 0
    total_after = 0

    for motif_name, info in sorted(results.items()):
        motif_key = _name_to_key(motif_name)
        clean_paths = info["clean"]

        # Count original images
        orig_n = info["n_total"]
        clean_n = len(clean_paths)

        if clean_n > 0:
            kg.seed_motif(motif_key, clean_paths)
            total_after += clean_n
        total_before += orig_n

        delta = clean_n - orig_n
        d_str = f"({delta:+d})" if delta != 0 else ""
        print(f"  {motif_name}: {orig_n}→{clean_n} {d_str}")

    st = kg.stats()
    print(f"\n  总计: {total_before}→{total_after} 张")
    print(f"  视觉质心: {st['motifs_with_visual_centroids']}/{st['motifs']}")

    return kg


def print_cleaning_report(results: dict):
    """Print cleaning summary."""
    total_clean = sum(r["n_clean"] for r in results.values())
    total_all = sum(r["n_total"] for r in results.values())

    print(f"\n{'='*64}")
    print(f"VL种子图像精化 — 汇总")
    print(f"{'='*64}")
    print(f"  总图像: {total_all}")
    print(f"  精化后保留: {total_clean} ({total_clean/total_all:.0%})")
    print(f"  滤除: {total_all - total_clean}")
    print()

    # By percentage clean
    print(f"  按精化率:")
    for name, info in sorted(results.items(), key=lambda x: -x[1]["pct_clean"]):
        bar = "█" * int(info["pct_clean"] * 20)
        empty = "·" * (20 - len(bar))
        print(f"    {name:8s} {bar}{empty} {info['n_clean']}/{info['n_total']}")

    # Common reasons for rejection
    print(f"\n  主要滤除原因 (目标纹样非Top-1):")
    confusions = defaultdict(int)
    for name, info in results.items():
        for d in info.get("details", []):
            if not d["is_clean"] and d["top1"]:
                confusions[f"{name}→{d['top1']}"] += 1
    for pair, count in sorted(confusions.items(), key=lambda x: -x[1])[:10]:
        print(f"    {pair}: {count}次")

    print(f"\n{'='*64}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean seed images using VL")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report, don't re-seed")
    args = parser.parse_args()

    print("=" * 64)
    print("VL种子图像精化")
    print("=" * 64)

    recognizer = VLMotifRecognizer()

    # Phase 1: Scan and classify
    print(f"\n[1] 扫描所有纹样目录 (阈值: {CONFIDENCE_THRESHOLD})")
    results = scan_and_clean(recognizer)

    # Phase 2: Report
    print_cleaning_report(results)

    if args.dry_run:
        # Save report
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "seed_cleaning.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            clean_results = {}
            for k, v in results.items():
                clean_results[k] = {
                    "n_clean": v["n_clean"],
                    "n_total": v["n_total"],
                    "pct_clean": v["pct_clean"],
                    "details": v["details"],
                }
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        print(f"报告已保存: {out_path}")
        return

    # Phase 3: Re-seed
    kg = reseed_with_clean(results)

    # Phase 4: Quick validation
    print(f"\n[2] 快速检索验证")
    for motif_name in ["凤凰纹", "龙纹", "万字纹", "回纹", "联珠纹"]:
        if motif_name in results and results[motif_name]["n_clean"] > 0:
            motif_key = _name_to_key(motif_name)
            desc = ATOMIC_MOTIFS.get(motif_key, {}).get("description", "")
            ranked = kg.query_text(desc[:80], top_k=5)
            rank = next((i+1 for i, (mid, _) in enumerate(ranked) if mid == motif_key), None)
            print(f"  {motif_name}: rank={rank}" if rank else f"  {motif_name}: not in top-5")

    print(f"\n{'='*64}")
    print("精化完成。运行 python -m src.evaluate_vl --compare 评估。")
    print(f"{'='*64}")


def _name_to_key(name: str) -> str:
    mapping = {
        "凤凰纹": "phoenix", "龙纹": "dragon", "蝴蝶纹": "butterfly",
        "麒麟纹": "qilin", "仙鹤纹": "crane", "鸳鸯纹": "mandarin_duck",
        "孔雀纹": "peacock", "鹿纹": "deer", "狮纹": "lion",
        "卷草纹": "scroll_grass", "牡丹纹": "peony", "莲花纹": "lotus",
        "宝相花纹": "baoxiang_flower", "梅花纹": "plum_blossom",
        "云纹": "cloud", "万字纹": "swastika", "回纹": "meander",
        "龟背纹": "turtle_shell", "联珠纹": "pearl_roundel",
        "联珠纹": "pearl_roundel",
        "文字纹": "auspicious_char",
    }
    return mapping.get(name, name)


if __name__ == "__main__":
    main()
