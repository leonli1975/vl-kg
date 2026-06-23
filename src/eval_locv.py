"""Leave-One-Class-Out cross-validation for VL motif recognition.

Addresses n=21 limitation by testing EVERY seed image (not just 1 per class),
yielding n≈155 test cases for more robust statistical conclusions.

Usage:
  python -m src.eval_locv           # full LOOCV (VL + dual-index)
  python -m src.eval_locv --vl-only  # VL only (faster, skip dual-index)
  python -m src.eval_locv --dry-run  # count test cases without running
"""

import json, time, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest

from .vl_recognizer import VLMotifRecognizer
from .enhanced_kg import EnhancedThreeLayerKG
from .evaluate_vl import DIR_TO_NAME, MOTIFS_DIR, _name_to_key

RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval_results"


def run_locv(vl_only: bool = False):
    """Run LOOCV: test every image, seed on the rest.

    Args:
        vl_only: if True, skip dual-index re-ranking (faster)
    """
    print("=" * 64)
    print("Leave-One-Class-Out Cross-Validation (LOOCV)")
    print("=" * 64)

    recognizer = VLMotifRecognizer()
    kg = EnhancedThreeLayerKG() if not vl_only else None

    results = []
    total_tests = 0
    vl_top1 = 0
    vl_top3 = 0
    re_top1 = 0
    re_top3 = 0
    category_vl = defaultdict(lambda: {"top1": 0, "top3": 0, "total": 0})
    category_re = defaultdict(lambda: {"top1": 0, "top3": 0, "total": 0})
    improvements = 0
    degradations = 0

    for dir_path in sorted(MOTIFS_DIR.iterdir()):
        if not dir_path.is_dir():
            continue
        dir_name = dir_path.name
        gt_name = DIR_TO_NAME.get(dir_name)
        if not gt_name:
            continue
        motif_key = _name_to_key(gt_name)

        # Get all images
        images = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            images.extend(sorted(dir_path.glob(ext)))
        if len(images) < 2:
            continue  # need at least 2 for CV

        from .three_layer_kg import ATOMIC_MOTIFS
        category = ATOMIC_MOTIFS.get(motif_key, {}).get("category", "?")

        n_correct_vl = 0
        n_correct_re = 0
        n_correct_vl3 = 0
        n_correct_re3 = 0

        for i, test_img in enumerate(images):
            test_path = str(test_img)
            seed_paths = [str(p) for j, p in enumerate(images) if j != i]
            if len(seed_paths) < 1:
                continue

            total_tests += 1

            # --- VL Recognition ---
            try:
                vl_result = recognizer.decompose_motifs(test_path)
                motifs = vl_result.get("motifs", [])
                vl_names = [m.get("name", "") for m in motifs]
                vl_confs = [m.get("confidence", 0.5) for m in motifs]
            except Exception as e:
                print(f"  ✗ VL error: {gt_name}/{test_img.name}: {e}")
                continue

            vl_top1_ok = (len(vl_names) > 0 and vl_names[0] == gt_name)
            vl_top3_ok = gt_name in vl_names[:3]

            if vl_top1_ok:
                n_correct_vl += 1
                vl_top1 += 1
            if vl_top3_ok:
                n_correct_vl3 += 1
                vl_top3 += 1

            # --- Dual-Index Re-Ranking ---
            re_top1_ok = vl_top1_ok  # default
            re_top3_ok = vl_top3_ok

            if not vl_only and kg is not None:
                try:
                    # Seed with remaining images
                    kg.seed_motif(motif_key, seed_paths)
                    r_feat = kg.extract_resnet(test_path)
                    c_feat = kg.extract_clip_vis(test_path)
                except Exception:
                    r_feat = None
                    c_feat = None

                if r_feat is not None:
                    # Re-rank VL top-3
                    vl_top3_list = list(zip(vl_names[:3], vl_confs[:3] + [0.0]*3))[:3]

                    scored = []
                    for cand_name, vl_conf in vl_top3_list:
                        cand_key = _name_to_key(cand_name)
                        store = kg.motif_embeddings.get(cand_key, {})

                        r_cent = store.get("resnet_centroid")
                        r_sim = 0.0
                        if r_cent is not None:
                            r_sim = np.dot(r_feat, r_cent) / (
                                np.linalg.norm(r_feat) * np.linalg.norm(r_cent) + 1e-8)

                        c_cent = store.get("clip_vis_centroid")
                        c_sim = 0.0
                        if c_cent is not None:
                            c_sim = np.dot(c_feat, c_cent) / (
                                np.linalg.norm(c_feat) * np.linalg.norm(c_cent) + 1e-8)

                        vis_sim = max(0, float((r_sim + c_sim) / 2))
                        blended = 0.6 * vis_sim + 0.4 * vl_conf
                        scored.append((cand_name, blended))

                    scored.sort(key=lambda x: x[1], reverse=True)
                    re_top1_name = scored[0][0] if scored else None
                    re_top3_names = [s[0] for s in scored[:3]]

                    re_top1_ok = (re_top1_name == gt_name)
                    re_top3_ok = gt_name in re_top3_names

            if re_top1_ok:
                n_correct_re += 1
                re_top1 += 1
            if re_top3_ok:
                n_correct_re3 += 1
                re_top3 += 1

            if not vl_top1_ok and re_top1_ok:
                improvements += 1
            elif vl_top1_ok and not re_top1_ok:
                degradations += 1

            results.append({
                "motif": gt_name,
                "category": category,
                "image": test_img.name,
                "vl_top1_ok": vl_top1_ok,
                "vl_top3_ok": vl_top3_ok,
                "re_top1_ok": re_top1_ok,
                "re_top3_ok": re_top3_ok,
                "vl_names": vl_names[:3],
            })

        n = len(images)
        print(f"  {gt_name}: VL={n_correct_vl}/{n} top1, Re={n_correct_re}/{n} top1 "
              f"(VL3={n_correct_vl3}/{n}, Re3={n_correct_re3}/{n})")

        category_vl[category]["top1"] += n_correct_vl
        category_vl[category]["top3"] += n_correct_vl3
        category_vl[category]["total"] += n
        category_re[category]["top1"] += n_correct_re
        category_re[category]["top3"] += n_correct_re3
        category_re[category]["total"] += n

    # --- Aggregate ---
    vl_acc = vl_top1 / total_tests if total_tests else 0
    vl3_acc = vl_top3 / total_tests if total_tests else 0
    re_acc = re_top1 / total_tests if total_tests else 0
    re3_acc = re_top3 / total_tests if total_tests else 0

    # McNemar test
    n_improved = improvements
    n_degraded = degradations
    n_discordant = n_improved + n_degraded
    mcnemar_p = binomtest(n_improved, n=n_discordant, p=0.5, alternative='greater').pvalue if n_discordant > 0 else 1.0

    print(f"\n{'='*64}")
    print(f"LOOCV Results (n={total_tests})")
    print(f"{'='*64}")
    print(f"  VL-only:          Top-1={vl_top1}/{total_tests} ({vl_acc:.1%})  "
          f"Top-3={vl_top3}/{total_tests} ({vl3_acc:.1%})")
    if not vl_only:
        print(f"  VL+Dual-Index:    Top-1={re_top1}/{total_tests} ({re_acc:.1%})  "
              f"Top-3={re_top3}/{total_tests} ({re3_acc:.1%})")
        print(f"  Δ Top-1:           {re_acc - vl_acc:+.1%}")
        print(f"  Improved: {n_improved}, Degraded: {n_degraded}")
        print(f"  McNemar p:         {mcnemar_p:.4f} "
              f"({'SIGNIFICANT (p<0.05)' if mcnemar_p < 0.05 else 'NOT significant'})")
    print(f"  Per category:")
    for cat in ["动物纹", "植物纹", "几何纹", "自然纹", "文字纹"]:
        cv = category_vl.get(cat, {})
        cr = category_re.get(cat, {})
        if cv.get("total", 0) > 0:
            t = cv["total"]
            vl1 = cv["top1"] / t if t else 0
            re1 = cr["top1"] / t if t else 0
            print(f"    {cat}: VL={vl1:.1%} ({cv['top1']}/{t}), "
                  f"Re={re1:.1%} ({cr['top1']}/{t})")

    agg = {
        "experiment": "LOOCV",
        "n_total": total_tests,
        "n_motifs": len(set(r["motif"] for r in results)),
        "vl_only": {
            "top1": vl_top1, "top3": vl_top3,
            "top1_acc": round(vl_acc, 4), "top3_acc": round(vl3_acc, 4),
        },
        "vl_rerank": {
            "top1": re_top1, "top3": re_top3,
            "top1_acc": round(re_acc, 4), "top3_acc": round(re3_acc, 4),
            "delta_top1": round(re_acc - vl_acc, 4),
            "improved": n_improved, "degraded": n_degraded,
            "mcnemar_p": round(float(mcnemar_p), 6),
            "significant_p05": bool(mcnemar_p < 0.05),
            "significant_p01": bool(mcnemar_p < 0.01),
        },
        "per_category": {
            str(cat): {
                "vl_top1": category_vl[cat]["top1"],
                "vl_top3": category_vl[cat]["top3"],
                "vl_total": category_vl[cat]["total"],
                "re_top1": category_re[cat]["top1"],
                "re_top3": category_re[cat]["top3"],
                "re_total": category_re[cat]["total"],
            }
            for cat in category_vl
        },
        "details": results,
    }


    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "locv_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")

    return agg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vl-only", action="store_true", help="VL only, skip dual-index")
    parser.add_argument("--dry-run", action="store_true", help="Count test cases only")
    args = parser.parse_args()

    if args.dry_run:
        total = 0
        for dir_path in sorted(MOTIFS_DIR.iterdir()):
            if not dir_path.is_dir():
                continue
            gt_name = DIR_TO_NAME.get(dir_path.name)
            if not gt_name:
                continue
            images = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
            if len(images) >= 2:
                print(f"  {gt_name}: {len(images)} images → {len(images)} test cases")
                total += len(images)
        print(f"\n  Total test cases: {total}")
        return

    run_locv(vl_only=args.vl_only)


if __name__ == "__main__":
    main()
