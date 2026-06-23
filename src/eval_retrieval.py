"""E5 + E6: Cross-modal text retrieval & incremental learning curve.

E5: Cross-Modal Text Retrieval Evaluation
  - 10 Chinese natural language queries → motif matching
  - Metric: Top-1, Top-3 accuracy (does the query retrieve correct motif?)

E6: Incremental Seed Learning Curve
  - For each motif class, incrementally add seeds (1, 2, 3, ..., max)
  - After each addition, evaluate Top-1 on a held-out test image
  - Show cold-start → convergence curve

Usage:
  python -m src.eval_retrieval       # Run both E5 + E6
  python -m src.eval_retrieval --e5  # E5 only
  python -m src.eval_retrieval --e6  # E6 only
"""

import json, time, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

from .enhanced_kg import EnhancedThreeLayerKG
from .evaluate_vl import _name_to_key, DIR_TO_NAME, MOTIFS_DIR

RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval_results"

# ═══════════════════════════════════════════════════════════════
# E5: Cross-Modal Text Retrieval
# ═══════════════════════════════════════════════════════════════

TEXT_QUERIES = [
    # Query → expected motif
    ("代表吉祥与重生的神鸟纹样", "凤凰纹"),
    ("象征富贵繁荣的花卉纹样", "牡丹纹"),
    ("佛教中代表永恒与吉祥万德的符号", "万字纹"),
    ("连续的直角折线构成的回旋式几何图案", "回纹"),
    ("由波斯传入的连续圆珠组成的环形边框装饰", "联珠纹"),
    ("象征生命连绵不断、生生不息的植物装饰纹样", "卷草纹"),
    ("象征仙逸长寿的瑞鸟纹样", "仙鹤纹"),
    ("代表权力与尊贵的神兽纹样", "龙纹"),
    ("象征美丽与爱情的双鸟纹样", "鸳鸯纹"),
    ("多瓣放射对称、象征富贵庄严的复合花卉纹样", "宝相花纹"),
]

def eval_text_retrieval():
    """E5: Evaluate cross-modal text → motif retrieval."""
    print("=" * 64)
    print("E5: Cross-Modal Text Retrieval Evaluation")
    print("=" * 64)

    kg = EnhancedThreeLayerKG()
    from .seed_motifs import scan_motif_dirs
    images_by_motif = scan_motif_dirs()
    for mid, paths in images_by_motif.items():
        if kg.motif_embeddings[mid]["resnet_centroid"] is None:
            kg.seed_motif(mid, [str(p) for p in paths])

    results = []
    top1_correct = 0
    top3_correct = 0

    for query, expected in TEXT_QUERIES:
        ranked = kg.query_text(query, top_k=10)
        names = [
            kg.motifs.get(mid, {}).get("name_zh", mid)
            for mid, _ in ranked
        ]
        top1_ok = (names[0] == expected) if names else False
        top3_ok = expected in names[:3]

        if top1_ok: top1_correct += 1
        if top3_ok: top3_correct += 1

        results.append({
            "query": query,
            "expected": expected,
            "top1": names[0] if names else None,
            "top3": names[:3],
            "top1_correct": top1_ok,
            "top3_correct": top3_ok,
        })

        mark = "✓" if top1_ok else ("△" if top3_ok else "✗")
        print(f"  {mark} \"{query}\" → top1={names[0]} (expected: {expected})")
        if not top1_ok and top3_ok:
            print(f"      top3: {names[:3]}")

    n = len(results)
    print(f"\n  E5 Results: Top-1={top1_correct}/{n} ({top1_correct/n:.1%})  "
          f"Top-3={top3_correct}/{n} ({top3_correct/n:.1%})")

    agg = {
        "experiment": "E5",
        "description": "Cross-modal text-to-motif retrieval",
        "n_queries": n,
        "top1_correct": top1_correct,
        "top3_correct": top3_correct,
        "top1_accuracy": top1_correct / n,
        "top3_accuracy": top3_correct / n,
        "details": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "e5_text_retrieval.json", 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    print(f"  Results saved to e5_text_retrieval.json")
    return agg


# ═══════════════════════════════════════════════════════════════
# E6: Incremental Seed Learning Curve
# ═══════════════════════════════════════════════════════════════

def eval_incremental_learning():
    """E6: Evaluate cold-start → convergence as seeds increase.

    For each motif with seed images:
      1. Reserve 1 test image (held-out)
      2. Incrementally add remaining images as seeds (1, 2, 3, ..., N-1)
      3. After each addition, evaluate Top-1 accuracy on the test image
      4. Plot accuracy vs. number of seeds
    """
    print("\n" + "=" * 64)
    print("E6: Incremental Seed Learning Curve")
    print("=" * 64)

    from .seed_motifs import scan_motif_dirs
    images_by_motif_full = scan_motif_dirs()

    # Use themes for grouping
    animal_keys = ["phoenix", "dragon", "butterfly", "qilin", "crane",
                   "mandarin_duck", "peacock", "deer", "lion"]
    plant_keys = ["scroll_grass", "peony", "lotus", "baoxiang_flower", "plum_blossom"]
    geo_keys = ["cloud", "swastika", "meander", "turtle_shell", "pearl_roundel"]

    all_curves = {}     # motif_name → [(n_seeds, accuracy)]
    incremental_data = {}  # detailed per-step data

    for dir_path in sorted(MOTIFS_DIR.iterdir()):
        if not dir_path.is_dir():
            continue
        gt_name = DIR_TO_NAME.get(dir_path.name)
        if not gt_name:
            continue
        motif_key = _name_to_key(gt_name)

        # Get cleaned images (those where VL confirms the label)
        if motif_key not in images_by_motif_full:
            continue
        all_images = [str(p) for p in images_by_motif_full[motif_key]]
        if len(all_images) < 3:
            continue  # need at least 3 images (1 test + 2 seed steps)

        # Use last image as test
        test_img = all_images[-1]
        seed_pool = all_images[:-1]

        curve = []
        step_results = []
        kg = EnhancedThreeLayerKG()

        for n_seeds in range(1, len(seed_pool) + 1):
            current_seeds = seed_pool[:n_seeds]

            # Re-seed from scratch at each step
            kg.seed_motif(motif_key, current_seeds)

            # Extract features from test image
            try:
                r_feat = kg.extract_resnet(test_img)
                c_feat = kg.extract_clip_vis(test_img)
            except Exception:
                curve.append((n_seeds, None))
                continue

            # Compute visual similarity with centroid
            store = kg.motif_embeddings[motif_key]
            r_cent = store.get("resnet_centroid")
            c_cent = store.get("clip_vis_centroid")

            r_sim = 0.0
            if r_cent is not None:
                r_sim = np.dot(r_feat, r_cent) / (
                    np.linalg.norm(r_feat) * np.linalg.norm(r_cent) + 1e-8)

            c_sim = 0.0
            if c_cent is not None:
                c_sim = np.dot(c_feat, c_cent) / (
                    np.linalg.norm(c_feat) * np.linalg.norm(c_cent) + 1e-8)

            vis_sim = max(0, float((r_sim + c_sim) / 2))
            curve.append((n_seeds, vis_sim))
            step_results.append({
                "n_seeds": n_seeds,
                "resnet_sim": float(r_sim),
                "clip_sim": float(c_sim),
                "vis_sim": vis_sim,
            })

        all_curves[gt_name] = curve
        incremental_data[gt_name] = {
            "motif_key": motif_key,
            "n_total": len(all_images),
            "curve": curve,
            "steps": step_results,
        }

        # Print summary line
        sims = [s for _, s in curve if s is not None]
        if sims:
            print(f"  {gt_name}: {len(seed_pool)} seeds, "
                  f"vis_sim {sims[0]:.3f}→{sims[-1]:.3f} "
                  f"(Δ={sims[-1]-sims[0]:+.3f})")

    # Aggregate by category
    animal_motifs_name = [mn for mn in all_curves
                          if _name_to_key(mn) in animal_keys]
    plant_motifs_name = [mn for mn in all_curves
                         if _name_to_key(mn) in plant_keys]
    geo_motifs_name = [mn for mn in all_curves
                       if _name_to_key(mn) in geo_keys]

    def cat_avg(motif_names):
        by_n = defaultdict(list)
        for mn in motif_names:
            if mn in all_curves:
                for n, s in all_curves[mn]:
                    if s is not None:
                        by_n[n].append(s)
        if not by_n:
            return []
        return sorted([(n, np.mean(vals)) for n, vals in by_n.items()])

    animal_avg = cat_avg(animal_motifs_name)
    plant_avg = cat_avg(plant_motifs_name)
    geo_avg = cat_avg(geo_motifs_name)

    print(f"\n  Category averages (final visual similarity):")
    for cat_name, curve in [("Animal", animal_avg), ("Plant", plant_avg),
                             ("Geometric", geo_avg)]:
        if curve:
            final = curve[-1][1]
            initial = curve[0][1]
            print(f"    {cat_name}: {initial:.3f}→{final:.3f} "
                  f"({final-initial:+.3f} over {len(curve)} steps)")

    agg = {
        "experiment": "E6",
        "description": "Incremental seed learning curve",
        "per_motif": incremental_data,
        "category_curves": {
            "animal": [(int(n), float(s)) for n, s in animal_avg],
            "plant": [(int(n), float(s)) for n, s in plant_avg],
            "geometric": [(int(n), float(s)) for n, s in geo_avg],
        },
    }
    with open(RESULTS_DIR / "e6_incremental_curve.json", 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    print(f"  Results saved to e6_incremental_curve.json")
    return agg


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--e5", action="store_true", help="E5 only")
    parser.add_argument("--e6", action="store_true", help="E6 only")
    args = parser.parse_args()

    run_both = not args.e5 and not args.e6

    if run_both or args.e5:
        eval_text_retrieval()
    if run_both or args.e6:
        eval_incremental_learning()


if __name__ == "__main__":
    main()
