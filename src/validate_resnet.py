"""Zero-shot validation using ResNet18 features (CLIP unavailable due to network)."""

import os
from pathlib import Path
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from collections import defaultdict

CLEAN_DIR = Path(__file__).parent.parent / "data" / "clean"

def load_model():
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Identity()  # remove classifier, keep 512-dim features
    model.eval()
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess

def extract_features(model, preprocess, image_paths):
    features = []
    valid_paths = []
    for p in image_paths:
        try:
            img = Image.open(p).convert('RGB')
            t = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                f = model(t).squeeze(0).numpy()
            features.append(f)
            valid_paths.append(p)
        except:
            pass
    return np.array(features), valid_paths

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def main():
    print("=" * 60)
    print("ResNet18 特征验证 — 蜀绣蜀锦图像类别可分性")
    print("=" * 60)

    model, preprocess = load_model()
    print("[1] ResNet18 已加载 (512-dim features)")

    # Collect all images and their categories
    all_paths = []
    all_labels = []
    cat_counts = {}
    for cat_dir in sorted(CLEAN_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.jpeg")) + list(cat_dir.glob("*.png"))
        for img in images[:10]:  # max 10 per category
            all_paths.append(str(img))
            all_labels.append(cat_dir.name)
        cat_counts[cat_dir.name] = min(len(images), 10)

    print(f"\n[2] 提取特征: {len(all_paths)} 张图像, {len(cat_counts)} 个类别")
    features, valid_paths = extract_features(model, preprocess, all_paths)
    valid_labels = [all_labels[all_paths.index(p)] for p in valid_paths]
    print(f"   有效特征: {len(features)}")

    # Compute intra-class vs inter-class similarity
    print(f"\n[3] 类内 vs 类间相似度分析")
    categories = sorted(set(valid_labels))
    
    # Per-category centroid
    centroids = {}
    for cat in categories:
        mask = [i for i, l in enumerate(valid_labels) if l == cat]
        if mask:
            centroids[cat] = np.mean(features[mask], axis=0)
    
    # Intra-class similarity
    intra_sims = []
    for cat in categories:
        mask = [i for i, l in enumerate(valid_labels) if l == cat]
        if len(mask) < 2:
            continue
        cat_feats = features[mask]
        for i in range(len(cat_feats)):
            for j in range(i+1, len(cat_feats)):
                intra_sims.append(cosine_sim(cat_feats[i], cat_feats[j]))
    
    # Inter-class similarity
    inter_sims = []
    for ci, cat_i in enumerate(categories):
        for cat_j in categories[ci+1:]:
            if cat_i not in centroids or cat_j not in centroids:
                continue
            inter_sims.append(cosine_sim(centroids[cat_i], centroids[cat_j]))
    
    intra_mean = np.mean(intra_sims) if intra_sims else 0
    inter_mean = np.mean(inter_sims) if inter_sims else 0
    
    print(f"   类内平均余弦相似度: {intra_mean:.4f}")
    print(f"   类间平均余弦相似度: {inter_mean:.4f}")
    print(f"   可分性指标 (intra - inter): {intra_mean - inter_mean:+.4f}")
    if intra_mean > inter_mean + 0.05:
        print(f"   ✓ 类内相似度显著高于类间 — 图像特征具有类别可分性")
    else:
        print(f"   ⚠ 类内/类间相似度差异不足 — 特征区分力有限")

    # Nearest-centroid classification
    print(f"\n[4] 最近质心分类准确率")
    correct = 0
    total = 0
    per_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for i, (feat, label) in enumerate(zip(features, valid_labels)):
        best_cat = None
        best_sim = -1
        for cat, cent in centroids.items():
            # Leave-one-out: exclude current sample from its own centroid
            if cat == label:
                mask = [j for j, l in enumerate(valid_labels) if l == cat and j != i]
                if len(mask) == 0:
                    continue
                cent = np.mean(features[mask], axis=0)
            sim = cosine_sim(feat, cent)
            if sim > best_sim:
                best_sim = sim
                best_cat = cat
        
        if best_cat == label:
            correct += 1
        total += 1
        per_cat[label]["total"] += 1
        if best_cat == label:
            per_cat[label]["correct"] += 1
    
    acc = correct / total if total > 0 else 0
    print(f"   总体准确率: {correct}/{total} = {acc:.1%}")
    print(f"\n   各类准确率:")
    for cat in sorted(per_cat.keys()):
        c = per_cat[cat]
        cat_short = cat.replace("shujin_", "锦·").replace("shuxiu_", "绣·")
        print(f"   {cat_short:<20s}: {c['correct']}/{c['total']} ({c['correct']/c['total']:.0%})")

    # Brocade vs Embroidery separability
    print(f"\n[5] 蜀锦 vs 蜀绣 质心距离")
    brocade_cats = [c for c in categories if c.startswith("shujin")]
    embroidery_cats = [c for c in categories if c.startswith("shuxiu")]
    
    if brocade_cats and embroidery_cats:
        brocade_feats = np.concatenate([features[[i for i,l in enumerate(valid_labels) if l==c]] for c in brocade_cats])
        embroidery_feats = np.concatenate([features[[i for i,l in enumerate(valid_labels) if l==c]] for c in embroidery_cats])
        
        b_cent = np.mean(brocade_feats, axis=0)
        e_cent = np.mean(embroidery_feats, axis=0)
        dist = np.linalg.norm(b_cent - e_cent)
        
        b_intra = np.mean([cosine_sim(f, b_cent) for f in brocade_feats])
        e_intra = np.mean([cosine_sim(f, e_cent) for f in embroidery_feats])
        be_inter = cosine_sim(b_cent, e_cent)
        
        print(f"   蜀锦类内相似度: {b_intra:.4f}")
        print(f"   蜀绣类内相似度: {e_intra:.4f}")
        print(f"   锦-绣间相似度:   {be_inter:.4f}")
        print(f"   质心欧氏距离:    {dist:.4f}")

    print(f"\n{'='*60}")
    print(f"[结论]")
    if acc > 1/len(categories) + 0.1:
        print(f"  ✓ 最近质心分类 ({acc:.1%}) 显著高于随机基线 "
              f"({1/len(categories):.1%})")
        print(f"  → ResNet18 特征具有类别区分力,VL模型方向值得推进")
    else:
        print(f"  ⚠ 分类准确率 ({acc:.1%}) 接近随机 — 可能需要更强的特征提取器")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
