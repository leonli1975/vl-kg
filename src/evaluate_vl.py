"""Batch evaluation of Qwen2.5-VL zero-shot motif recognition.

Evaluates VL recognition accuracy on the 21 seeded motif directories.
Each directory name = ground truth label.

Modes:
  --vl-only:    pure VL zero-shot (default)
  --rerank:     VL + dual-index (ResNet+CN-CLIP) re-ranking
  --compare:    run both and compare VL-only vs VL+dual-index

Metrics:
  - Top-1 accuracy: ground truth is the #1 recognized motif
  - Top-3 accuracy: ground truth is in top 3 recognized motifs
  - Per-category breakdown
  - Per-motif detailed results

Usage:
  python -m src.evaluate_vl                    # VL-only evaluation
  python -m src.evaluate_vl --rerank            # VL + dual-index re-rank
  python -m src.evaluate_vl --compare           # A/B comparison
  python -m src.evaluate_vl --samples 2         # 2 images per motif
"""

import json, sys, time, numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

from .vl_recognizer import VLMotifRecognizer
from .three_layer_kg import ATOMIC_MOTIFS
from .enhanced_kg import EnhancedThreeLayerKG

MOTIFS_DIR = Path(__file__).parent.parent / "data" / "images" / "motifs"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval_results"

# Mapping: directory name → ground truth motif name
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


def evaluate_motif_images(samples_per_motif: int = 2,
                          category_filter: str = None) -> dict:
    """Run VL recognition on seed images and evaluate accuracy.

    Args:
        samples_per_motif: number of images to test per motif
        category_filter: only evaluate motifs in this category (e.g., '动物纹')

    Returns:
        evaluation results dict with per-motif and aggregate metrics
    """
    recognizer = VLMotifRecognizer()
    results = {
        "config": {
            "model": recognizer.model,
            "samples_per_motif": samples_per_motif,
            "category_filter": category_filter,
        },
        "per_motif": {},
        "aggregate": {},
    }

    total_top1 = 0
    total_top3 = 0
    total_tests = 0
    category_stats = defaultdict(lambda: {"top1": 0, "top3": 0, "total": 0})

    # Iterate over motif directories
    for dir_path in sorted(MOTIFS_DIR.iterdir()):
        if not dir_path.is_dir():
            continue
        dir_name = dir_path.name
        gt_name = DIR_TO_NAME.get(dir_name)
        if not gt_name:
            continue

        # Get category
        from .three_layer_kg import ATOMIC_MOTIFS
        motif_key = _name_to_key(gt_name)
        category = ATOMIC_MOTIFS.get(motif_key, {}).get("category", "?")

        if category_filter and category != category_filter:
            continue

        # Find image files
        images = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            images.extend(sorted(dir_path.glob(ext)))
        images = images[:samples_per_motif]

        if not images:
            continue

        motif_results = []
        motif_top1 = 0
        motif_top3 = 0

        for img_path in images:
            print(f"  [{category}] {gt_name}: {img_path.name}...", end=" ", flush=True)
            try:
                vl_result = recognizer.decompose_motifs(str(img_path))
                motifs = vl_result.get("motifs", [])

                # Check if ground truth is in results
                names = [m.get("name", "") for m in motifs]
                top1_correct = (len(names) > 0 and names[0] == gt_name)
                top3_correct = gt_name in names[:3]

                status = "✓" if top1_correct else ("△" if top3_correct else "✗")
                elapsed = vl_result.get("_elapsed", 0)

                motif_results.append({
                    "image": img_path.name,
                    "ground_truth": gt_name,
                    "top1_name": names[0] if names else None,
                    "top1_correct": top1_correct,
                    "top3_names": names[:3],
                    "top3_correct": top3_correct,
                    "all_names": names,
                    "vl_elapsed": elapsed,
                })

                print(f"{status} top1={names[0] if names else '?'} "
                      f"({elapsed:.0f}s)")

                if top1_correct:
                    motif_top1 += 1
                    total_top1 += 1
                if top3_correct:
                    motif_top3 += 1
                    total_top3 += 1
                total_tests += 1

            except Exception as e:
                print(f"✗ ERROR: {e}")
                motif_results.append({
                    "image": img_path.name,
                    "ground_truth": gt_name,
                    "error": str(e),
                })

        n = len(images)
        cat_stats = category_stats[category]
        cat_stats["top1"] += motif_top1
        cat_stats["top3"] += motif_top3
        cat_stats["total"] += n

        results["per_motif"][gt_name] = {
            "category": category,
            "n_images": n,
            "top1_correct": motif_top1,
            "top3_correct": motif_top3,
            "top1_acc": motif_top1 / n if n else 0,
            "top3_acc": motif_top3 / n if n else 0,
            "details": motif_results,
        }

        acc_str = f"{motif_top1}/{n} top1, {motif_top3}/{n} top3"
        print(f"    → {acc_str}")

    # ── Aggregate ──
    results["aggregate"] = {
        "total_tests": total_tests,
        "top1_correct": total_top1,
        "top3_correct": total_top3,
        "top1_accuracy": total_top1 / total_tests if total_tests else 0,
        "top3_accuracy": total_top3 / total_tests if total_tests else 0,
        "by_category": {
            cat: {
                "top1_acc": s["top1"] / s["total"] if s["total"] else 0,
                "top3_acc": s["top3"] / s["total"] if s["total"] else 0,
                "n": s["total"],
            }
            for cat, s in category_stats.items()
        },
        "n_motifs_evaluated": len(results["per_motif"]),
    }

    return results


def _name_to_key(name: str) -> str:
    """Map Chinese motif name to motif key."""
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


class DualIndexReranker:
    """Use ResNet+CN-CLIP centroids to re-rank VL Top-3 candidates."""

    def __init__(self):
        print("  加载双索引质心...")
        self.kg = EnhancedThreeLayerKG()

        # Seed all motifs with their images
        from .seed_motifs import scan_motif_dirs
        images_by_motif = scan_motif_dirs()
        for motif_id, img_paths in images_by_motif.items():
            if self.kg.motif_embeddings[motif_id]["resnet_centroid"] is None:
                self.kg.seed_motif(motif_id, [str(p) for p in img_paths])

        st = self.kg.stats()
        print(f"    {st['motifs_with_visual_centroids']}/{st['motifs']} 纹样有视觉质心")

        # Weights for blending
        self.alpha_vis = 0.6  # visual similarity weight
        self.alpha_vl = 0.4   # VL confidence weight

    def rerank(self, image_path: str, vl_top3: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Re-rank VL Top-3 candidates using dual-index visual similarity.

        Args:
            image_path: path to the test image
            vl_top3: [(motif_name, vl_confidence), ...]

        Returns:
            [(motif_name, blended_score), ...] sorted by blended score
        """
        # Extract features from test image
        try:
            r_feat = self.kg.extract_resnet(image_path)
            c_feat = self.kg.extract_clip_vis(image_path)
        except Exception:
            return vl_top3  # fallback: return VL order

        scored = []
        for motif_name, vl_conf in vl_top3:
            motif_key = _name_to_key(motif_name)
            store = self.kg.motif_embeddings.get(motif_key, {})

            # ResNet similarity
            r_cent = store.get("resnet_centroid")
            r_sim = 0.0
            if r_cent is not None:
                r_sim = np.dot(r_feat, r_cent) / (
                    np.linalg.norm(r_feat) * np.linalg.norm(r_cent) + 1e-8
                )
                r_sim = max(0, float(r_sim))  # cosine similarity

            # CLIP similarity
            c_cent = store.get("clip_vis_centroid")
            c_sim = 0.0
            if c_cent is not None:
                c_sim = np.dot(c_feat, c_cent) / (
                    np.linalg.norm(c_feat) * np.linalg.norm(c_cent) + 1e-8
                )
                c_sim = max(0, float(c_sim))

            # Visual similarity: average of ResNet + CLIP
            vis_sim = (r_sim + c_sim) / 2

            # Blended score
            blended = self.alpha_vis * vis_sim + self.alpha_vl * vl_conf
            scored.append((motif_name, blended, {
                "vl_conf": vl_conf,
                "resnet_sim": r_sim,
                "clip_sim": c_sim,
            }))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def evaluate_with_rerank(samples_per_motif: int = 1) -> dict:
    """Run VL recognition + dual-index re-ranking and compare.

    Returns comparison: VL-only vs VL+dual-index accuracy.
    """
    recognizer = VLMotifRecognizer()
    reranker = DualIndexReranker()

    results = {
        "vl_only": {"top1": 0, "top3": 0, "total": 0},
        "vl_rerank": {"top1": 0, "top3": 0, "total": 0},
        "per_motif": {},
        "improvements": [],
    }

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
        images = images[:samples_per_motif]
        if not images:
            continue

        motif_vl_top1 = 0
        motif_re_top1 = 0
        motif_vl_top3 = 0
        motif_re_top3 = 0
        motif_improved = 0

        for img_path in images:
            img_str = str(img_path)
            try:
                vl_result = recognizer.decompose_motifs(img_str)
                motifs = vl_result.get("motifs", [])

                # VL-only ranking
                vl_names = [m.get("name", "") for m in motifs]
                vl_confs = [m.get("confidence", 0.5) for m in motifs]

                # Pad to at least 3
                while len(vl_names) < 3:
                    vl_names.append(None)
                    vl_confs.append(0.0)

                vl_top3_list = list(zip(vl_names[:3], vl_confs[:3]))

                # VL-only accuracy
                vl_top1_ok = (vl_names[0] == gt_name)
                vl_top3_ok = gt_name in vl_names[:3]

                if vl_top1_ok:
                    motif_vl_top1 += 1
                if vl_top3_ok:
                    motif_vl_top3 += 1

                # VL + dual-index re-ranking
                re_ranked = reranker.rerank(img_str, vl_top3_list)
                re_top1_name = re_ranked[0][0] if re_ranked else None
                re_top3_names = [r[0] for r in re_ranked[:3]]

                re_top1_ok = (re_top1_name == gt_name)
                re_top3_ok = gt_name in re_top3_names

                if re_top1_ok:
                    motif_re_top1 += 1
                if re_top3_ok:
                    motif_re_top3 += 1

                # Track improvements
                improved = (not vl_top1_ok and re_top1_ok)
                if improved:
                    motif_improved += 1
                    results["improvements"].append({
                        "image": img_path.name,
                        "ground_truth": gt_name,
                        "vl_top1": vl_names[0],
                        "re_top1": re_top1_name,
                    })

                status = "✓" if vl_top1_ok else ("↑" if improved else "✗")
                print(f"  {status} {gt_name}: VL={vl_names[0]} "
                      f"→ Rerank={re_top1_name}")

            except Exception as e:
                print(f"  ✗ {gt_name}: ERROR - {e}")

        n = len(images)
        results["vl_only"]["top1"] += motif_vl_top1
        results["vl_only"]["top3"] += motif_vl_top3
        results["vl_only"]["total"] += n
        results["vl_rerank"]["top1"] += motif_re_top1
        results["vl_rerank"]["top3"] += motif_re_top3
        results["vl_rerank"]["total"] += n

        results["per_motif"][gt_name] = {
            "n": n,
            "vl_top1": motif_vl_top1,
            "vl_top3": motif_vl_top3,
            "re_top1": motif_re_top1,
            "re_top3": motif_re_top3,
            "improved": motif_improved,
        }

        delta = motif_re_top1 - motif_vl_top1
        d = f"+{delta}" if delta > 0 else str(delta)
        print(f"    → VL:{motif_vl_top1}/{n} Re:{motif_re_top1}/{n} (Δ={d})")

    # Compute accuracies
    t = results["vl_only"]["total"]
    if t > 0:
        results["vl_only"]["accuracy"] = results["vl_only"]["top1"] / t
        results["vl_rerank"]["accuracy"] = results["vl_rerank"]["top1"] / t
        results["delta_top1"] = results["vl_rerank"]["top1"] - results["vl_only"]["top1"]
        results["delta_accuracy"] = results["vl_rerank"]["accuracy"] - results["vl_only"]["accuracy"]

    return results


def print_comparison(results: dict):
    """Print A/B comparison report."""
    vl = results["vl_only"]
    re = results["vl_rerank"]

    print(f"\n{'='*64}")
    print(f"VL vs VL+双索引重排序 — 对比报告")
    print(f"{'='*64}")
    print(f"总测试数: {vl['total']}")
    print()
    print(f"  {'':20s} {'Top-1':>10s} {'Top-3':>10s}")
    print(f"  {'VL 仅':20s} {vl['top1']}/{vl['total']} ({vl['accuracy']:.1%})  "
          f"{vl['top3']}/{vl['total']} ({vl['top3']/vl['total']:.1%})")
    print(f"  {'VL + 双索引':20s} {re['top1']}/{re['total']} ({re['accuracy']:.1%})  "
          f"{re['top3']}/{re['total']} ({re['top3']/re['total']:.1%})")
    print(f"  {'提升':20s} +{results['delta_top1']} ({results['delta_accuracy']:+.1%})")

    # Per-motif improvements
    print(f"\n  每纹样改进:")
    per = results["per_motif"]
    improved_motifs = []
    for name, m in sorted(per.items()):
        if m["improved"] > 0:
            improved_motifs.append(name)
            print(f"    ↑ {name}: VL→{m['vl_top1']}/{m['n']} Re→{m['re_top1']}/{m['n']}")

    if not improved_motifs:
        print(f"    (无改进)")

    # Show specific fixes
    imp = results.get("improvements", [])
    if imp:
        print(f"\n  具体修正案例:")
        for i in imp[:10]:
            print(f"    {i['ground_truth']}: VL='{i['vl_top1']}' → Re='{i['re_top1']}' ({i['image']})")

    print(f"\n{'='*64}")


def print_report(results: dict):
    """Print formatted evaluation report."""
    agg = results["aggregate"]

    print(f"\n{'='*64}")
    print(f"VL 零样本纹样识别 — 评估报告")
    print(f"{'='*64}")
    print(f"模型: {results['config']['model']}")
    print(f"每纹样样本数: {results['config']['samples_per_motif']}")
    print(f"评估纹样数: {agg['n_motifs_evaluated']}")
    print(f"总测试数: {agg['total_tests']}")
    print()
    print(f"  Top-1 准确率: {agg['top1_correct']}/{agg['total_tests']} = {agg['top1_accuracy']:.1%}")
    print(f"  Top-3 准确率: {agg['top3_correct']}/{agg['total_tests']} = {agg['top3_accuracy']:.1%}")
    print()

    # By category
    print(f"  按类别:")
    for cat in ["动物纹", "植物纹", "几何纹", "自然纹", "文字纹"]:
        c = agg["by_category"].get(cat)
        if c:
            print(f"    {cat}: top1={c['top1_acc']:.1%} top3={c['top3_acc']:.1%} "
                  f"(n={c['n']})")

    # Per-motif detail
    print(f"\n  按纹样:")
    per = results["per_motif"]
    for name in sorted(per.keys()):
        m = per[name]
        top1_bar = "█" * int(m["top1_acc"] * 20)
        top3_bar = "░" * int((m["top3_acc"] - m["top1_acc"]) * 20)
        fail_bar = "·" * (20 - len(top1_bar) - len(top3_bar))
        print(f"    {name:8s} [{m['category']:4s}] "
              f"{top1_bar}{top3_bar}{fail_bar} "
              f"top1={m['top1_acc']:.0%} top3={m['top3_acc']:.0%}")

    # Common errors
    print(f"\n  常见误识别:")
    errors = defaultdict(int)
    for name, m in per.items():
        for d in m.get("details", []):
            if not d.get("top1_correct") and d.get("all_names"):
                for wrong in d["all_names"][:2]:
                    if wrong != name:
                        errors[f"{name}→{wrong}"] += 1
    for pair, count in sorted(errors.items(), key=lambda x: -x[1])[:10]:
        print(f"    {pair}: {count}次")

    print(f"\n{'='*64}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate VL motif recognition")
    parser.add_argument("--samples", type=int, default=1,
                        help="Images per motif (default: 1)")
    parser.add_argument("--category", help="Filter by category (e.g., 动物纹)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--rerank", action="store_true",
                        help="VL + dual-index re-ranking")
    parser.add_argument("--compare", action="store_true",
                        help="Run A/B comparison: VL-only vs VL+dual-index")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare or args.rerank:
        print("=" * 64)
        print("VL + 双索引重排序 评估")
        print("=" * 64)

        results = evaluate_with_rerank(samples_per_motif=args.samples)
        print_comparison(results)

        out_path = args.output or str(RESULTS_DIR / "vl_rerank_eval.json")
        # Convert numpy types for JSON
        clean = json.loads(json.dumps(results, default=str))
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {out_path}")
    else:
        print("=" * 64)
        print("Qwen2.5-VL 零样本纹样识别评估")
        print("=" * 64)

        results = evaluate_motif_images(
            samples_per_motif=args.samples,
            category_filter=args.category,
        )
        print_report(results)

        out_path = args.output or str(RESULTS_DIR / "vl_eval.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
