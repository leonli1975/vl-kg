"""CLIP zero-shot validation on Shu embroidery/brocade images."""

import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import open_clip

CLEAN_DIR = Path(__file__).parent.parent / "data" / "clean"

def load_model():
    """Load OpenCLIP ViT-B/32 (lightweight, works on CPU)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

def zero_shot_classify(model, preprocess, tokenizer, image_paths, class_labels):
    """Classify images using CLIP zero-shot image-text matching."""
    results = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0)
            
            text_tokens = tokenizer(class_labels)
            
            with torch.no_grad():
                image_features = model.encode_image(img_tensor)
                text_features = model.encode_text(text_tokens)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarities = (100.0 * image_features @ text_features.T).squeeze(0)
            
            best_idx = similarities.argmax().item()
            best_label = class_labels[best_idx]
            best_score = similarities[best_idx].item()
            top3_indices = similarities.topk(3).indices.tolist()
            top3 = [(class_labels[i], similarities[i].item()) for i in top3_indices]
            
            results.append({
                "path": str(img_path),
                "predicted": best_label,
                "score": best_score,
                "top3": top3,
            })
        except Exception as e:
            results.append({"path": str(img_path), "error": str(e)})
    return results

def main():
    print("=" * 60)
    print("CLIP 零样本验证 — 蜀绣蜀锦纹样识别")
    print("=" * 60)
    
    # Load model
    print("\n[1] 加载 OpenCLIP ViT-B/32...")
    model, preprocess, tokenizer = load_model()
    print("   模型加载完成")
    
    # ---- Test 1: Category-level (brocade vs embroidery) ----
    print("\n[2] 测试1: 蜀锦 vs 蜀绣 大类区分")
    cat_labels = [
        "a traditional Chinese Shu brocade woven textile with geometric patterns",
        "a traditional Chinese Shu embroidery hand-stitched textile with floral patterns",
        "a modern photograph or unrelated image",
    ]
    
    # Collect images from different categories
    test_images = []
    for cat_dir in sorted(CLEAN_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.jpeg")) + list(cat_dir.glob("*.png"))
        test_images.extend(images[:3])  # 3 per category
    
    print(f"   测试图像: {len(test_images)} 张")
    results = zero_shot_classify(model, preprocess, tokenizer, test_images, cat_labels)
    
    # Summarize
    cat_map = {0: "蜀锦", 1: "蜀绣", 2: "其他"}
    correct = 0
    for r in results:
        if "error" in r:
            continue
        # Determine ground truth from path
        cat_name = Path(r["path"]).parent.name
        is_brocade = cat_name.startswith("shujin")
        is_embroidery = cat_name.startswith("shuxiu")
        gt = 0 if is_brocade else (1 if is_embroidery else 2)
        pred_idx = cat_labels.index(r["predicted"])
        if pred_idx == gt:
            correct += 1
    
    acc = correct / len(results) if results else 0
    print(f"   大类区分准确率: {correct}/{len(results)} = {acc:.1%}")
    
    # Show some examples
    print(f"\n   示例预测:")
    for r in results[:5]:
        if "error" in r:
            continue
        cat = Path(r["path"]).parent.name
        print(f"   [{cat}] → {r['predicted'][:60]}... (score={r['score']:.1f})")
    
    # ---- Test 2: Pattern type (for brocade) ----
    print("\n[3] 测试2: 蜀锦纹样类型识别")
    pattern_labels = [
        "a Shu brocade textile with phoenix bird pattern",
        "a Shu brocade textile with cloud pattern (yunwen)",
        "a Shu brocade textile with scrolling grass pattern (juancaowen)",
        "a Shu brocade textile with moon halo pattern (yuehua)",
        "a Shu brocade textile with rain thread pattern (yusi)",
        "a Shu brocade textile with square brocade pattern (fangfang)",
    ]
    
    # Only test on brocade images with known categories
    pattern_results = {}
    for cat_dir in sorted(CLEAN_DIR.iterdir()):
        if not cat_dir.is_dir() or not cat_dir.name.startswith("shujin"):
            continue
        cat_name = cat_dir.name.replace("shujin_", "")
        images = list(cat_dir.glob("*.jpg"))[:5]
        if not images:
            continue
        
        res = zero_shot_classify(model, preprocess, tokenizer, images, pattern_labels)
        predictions = [r["predicted"] for r in res if "error" not in r]
        if predictions:
            # Map ground truth
            gt_map = {
                "phoenix": "phoenix bird pattern",
                "cloud": "cloud pattern",
                "scroll": "scrolling grass pattern",
                "yuehua": "moon halo pattern",
                "yusi": "rain thread pattern",
                "fangfang": "square brocade pattern",
            }
            gt_word = gt_map.get(cat_name, "")
            hit = sum(1 for p in predictions if gt_word and gt_word in p.lower())
            pattern_results[cat_name] = {
                "total": len(predictions),
                "correct": hit,
                "top_pred": max(set(predictions), key=predictions.count),
                "predictions": predictions,
            }
            print(f"   {cat_name}: {hit}/{len(predictions)} ({hit/len(predictions):.0%}) top={pattern_results[cat_name]['top_pred'][:50]}")
    
    # ---- Test 3: Stitch type (for embroidery) ----
    print("\n[4] 测试3: 蜀绣针法类型识别")
    stitch_labels = [
        "a Shu embroidery with flat stitch (pingxiu), smooth and even surface",
        "a Shu embroidery with seed stitch (dazi xiu), granular knot-like texture",
        "a Shu embroidery with random stitch (luanzhen xiu), short crisscross threads",
        "a Shu embroidery with halo stitch (yunzhen), gradient color transition",
    ]
    
    stitch_results = {}
    for cat_dir in sorted(CLEAN_DIR.iterdir()):
        if not cat_dir.is_dir() or not cat_dir.name.startswith("shuxiu"):
            continue
        cat_name = cat_dir.name.replace("shuxiu_", "")
        if cat_name == "general" or cat_name == "phoenix":
            continue  # skip general/theme categories
        images = list(cat_dir.glob("*.jpg"))[:5]
        if not images:
            continue
        
        res = zero_shot_classify(model, preprocess, tokenizer, images, stitch_labels)
        predictions = [r["predicted"] for r in res if "error" not in r]
        if predictions:
            gt_map = {
                "pingxiu": "flat stitch",
                "dazi": "seed stitch",
                "luanzhen": "random stitch",
                "yunzhen": "halo stitch",
            }
            gt_word = gt_map.get(cat_name, "")
            hit = sum(1 for p in predictions if gt_word and gt_word in p.lower())
            stitch_results[cat_name] = {
                "total": len(predictions),
                "correct": hit,
                "top_pred": max(set(predictions), key=predictions.count),
            }
            print(f"   {cat_name}: {hit}/{len(predictions)} ({hit/len(predictions):.0%}) top={stitch_results[cat_name]['top_pred'][:50]}")
    
    # ---- Summary ----
    print(f"\n{'='*60}")
    print("[总结]")
    print(f"  大类区分 (蜀锦/蜀绣/其他): {acc:.1%}")
    if pattern_results:
        brocade_acc = sum(r["correct"] for r in pattern_results.values()) / max(1, sum(r["total"] for r in pattern_results.values()))
        print(f"  蜀锦纹样类型: {brocade_acc:.1%}")
    if stitch_results:
        stitch_acc = sum(r["correct"] for r in stitch_results.values()) / max(1, sum(r["total"] for r in stitch_results.values()))
        print(f"  蜀绣针法类型: {stitch_acc:.1%}")
    
    print(f"\n  结论: CLIP零样本在蜀绣/蜀锦图像上的初步验证")
    if acc > 0.5:
        print(f"  ✓ 大类区分可行 (>{acc:.0%}) — VL模型方向值得继续推进")
    else:
        print(f"  ⚠ 大类区分准确率较低 — 可能需要微调或更专门的模型")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
