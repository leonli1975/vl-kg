"""Integrated prototype: CN-CLIP + ResNet18 dual-index enhanced KG.

Dual-index architecture:
  - ResNet18: pure visual similarity (fast, lightweight)
  - CN-CLIP:   cross-modal image↔text matching (semantic alignment)

KG stores:
  - Per-class: ResNet centroid + CLIP visual centroid + CLIP text embedding
  - Ontology:  pattern type, regularity, scale, visual description (from annotation spec)
"""

import os, time, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import cn_clip.clip.model as clip_model
import cn_clip.clip as clip

# ---- Config ----
DTD_DIR = Path("/home/lgc/OCtestprj/fabric_structure/prototype/data/dtd/images")
CN_CLIP_CKPT = "/home/lgc/.cache/modelscope/damo/multi-modal_clip-vit-base-patch16_zh/clip_cn_vit-b-16.pt"
FABRIC_CLASSES = [
    "woven", "knitted", "striped", "chequered", "dotted", "grid",
    "banded", "braided", "pleated", "paisley", "meshed", "polka-dotted",
]
SEED_PER_CLASS = 3
TEST_PER_CLASS = 10

# KG Ontology (from annotation spec — will be enriched by actual annotations)
PATTERN_ONTOLOGY = {
    "woven":     {"type": "机织纹理", "regularity": "高", "scale": "细密",
                  "visual": "正交经纬网格,紧密交织结构,类似平纹织物的规律纹理"},
    "knitted":   {"type": "针织纹理", "regularity": "高", "scale": "中等",
                  "visual": "V形环状线圈,弹性结构,类似手工编织的毛线纹理"},
    "striped":   {"type": "条纹纹理", "regularity": "高", "scale": "中等",
                  "visual": "平行带状条纹,交替色彩或纹理,纵向或横向排列"},
    "chequered": {"type": "格纹纹理", "regularity": "极高", "scale": "中等",
                  "visual": "规则棋盘格图案,对比色块交替排列"},
    "dotted":    {"type": "点状纹理", "regularity": "高", "scale": "细密",
                  "visual": "规则排列的圆形斑点,均匀分布在底色上"},
    "grid":      {"type": "网格纹理", "regularity": "极高", "scale": "粗大",
                  "visual": "规则矩形网格,类似窗户分隔,线条清晰"},
    "banded":    {"type": "层带状纹理", "regularity": "中等", "scale": "粗大",
                  "visual": "水平或垂直带状层叠,类似地质岩层的分层纹理"},
    "braided":   {"type": "编织纹理", "regularity": "高", "scale": "中等",
                  "visual": "对角线交织的辫状结构,类似发辫的交叉纹理"},
    "pleated":   {"type": "褶皱纹理", "regularity": "中等", "scale": "粗大",
                  "visual": "平行折叠产生的棱纹表面,类似百褶裙的褶皱效果"},
    "paisley":   {"type": "佩斯利纹样", "regularity": "低", "scale": "中等",
                  "visual": "弯曲的泪滴形佩斯利图案,华丽的涡卷花纹,类似传统印染纹样"},
    "meshed":    {"type": "网眼纹理", "regularity": "中等", "scale": "细密",
                  "visual": "互相连通的网孔结构,类似渔网或纱窗的网状纹理"},
    "polka-dotted": {"type": "圆点纹理", "regularity": "高", "scale": "粗大",
                  "visual": "醒目的圆形大斑点均匀分布,类似波尔卡圆点图案"},
}


class DualIndex:
    """Enhanced index with ResNet visual + CLIP cross-modal embeddings."""
    
    def __init__(self):
        self._init_resnet()
        self._init_clip()
        self.templates = {}
        for cls in FABRIC_CLASSES:
            self.templates[cls] = {
                "resnet_features": [],       # List of 512-dim vectors
                "resnet_centroid": None,
                "clip_vis_features": [],     # List of 512-dim vectors
                "clip_vis_centroid": None,
                "clip_text_emb": None,       # 512-dim text embedding from ontology
                "n": 0,
            }
    
    def _init_resnet(self):
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Identity(); model.eval()
        self.resnet = model
        self.resnet_prep = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    
    def _init_clip(self):
        ckpt = torch.load(CN_CLIP_CKPT, map_location='cpu', weights_only=True)
        model = clip_model.CLIP(
            embed_dim=512, image_resolution=224, vision_layers=12, vision_width=768,
            vision_patch_size=16, vocab_size=21128, text_attention_probs_dropout_prob=0.1,
            text_hidden_act='gelu', text_hidden_dropout_prob=0.1, text_hidden_size=768,
            text_initializer_range=0.02, text_intermediate_size=3072,
            text_max_position_embeddings=512, text_num_attention_heads=12,
            text_num_hidden_layers=12, text_type_vocab_size=2,
        )
        model.load_state_dict(ckpt, strict=False); model.eval()
        self.clip_model = model
        self.clip_prep = T.Compose([
            T.Resize(224, interpolation=Image.BICUBIC), T.CenterCrop(224), T.ToTensor(),
            T.Normalize((0.48145466,0.4578275,0.40821073), (0.26862954,0.26130258,0.27577711)),
        ])
        # Pre-compute CLIP text embeddings for ontology descriptions
        self.onto_text_embs = {}
        for cls, onto in PATTERN_ONTOLOGY.items():
            desc = onto["visual"]  # Use visual description as text query
            tokens = clip.tokenize([desc])
            with torch.no_grad():
                emb = self.clip_model.encode_text(tokens)
                emb /= emb.norm(dim=-1, keepdim=True)
            self.onto_text_embs[cls] = emb.squeeze(0).numpy()
    
    def extract_resnet(self, img_path):
        img = Image.open(img_path).convert('RGB')
        t = self.resnet_prep(img).unsqueeze(0)
        with torch.no_grad():
            return self.resnet(t).squeeze(0).numpy()
    
    def extract_clip_vis(self, img_path):
        img = Image.open(img_path).convert('RGB')
        t = self.clip_prep(img).unsqueeze(0)
        with torch.no_grad():
            f = self.clip_model.encode_image(t)
            f /= f.norm(dim=-1, keepdim=True)
        return f.squeeze(0).numpy()
    
    def seed(self, cls, img_paths):
        r_feats = [self.extract_resnet(p) for p in img_paths]
        c_feats = [self.extract_clip_vis(p) for p in img_paths]
        self.templates[cls]["resnet_features"] = r_feats
        self.templates[cls]["resnet_centroid"] = np.mean(r_feats, axis=0)
        self.templates[cls]["clip_vis_features"] = c_feats
        self.templates[cls]["clip_vis_centroid"] = np.mean(c_feats, axis=0)
        self.templates[cls]["n"] = len(img_paths)
        self.templates[cls]["clip_text_emb"] = self.onto_text_embs.get(cls)
    
    def query_visual(self, feature, top_k=3, use_clip=False):
        """Image→Class retrieval using visual features."""
        scores = {}
        feat_key = "clip_vis_centroid" if use_clip else "resnet_centroid"
        for cls, tpl in self.templates.items():
            cent = tpl.get(feat_key)
            if cent is None:
                continue
            sim = np.dot(feature, cent) / (np.linalg.norm(feature) * np.linalg.norm(cent) + 1e-8)
            scores[cls] = sim
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def query_text(self, text_query, top_k=3):
        """Text→Class retrieval using CLIP cross-modal matching."""
        tokens = clip.tokenize([text_query])
        with torch.no_grad():
            txt_emb = self.clip_model.encode_text(tokens)
            txt_emb /= txt_emb.norm(dim=-1, keepdim=True)
        txt_vec = txt_emb.squeeze(0).numpy()
        
        scores = {}
        # Match against CLIP visual centroids
        for cls, tpl in self.templates.items():
            if tpl["clip_vis_centroid"] is None:
                continue
            sim = np.dot(txt_vec, tpl["clip_vis_centroid"])
            scores[cls] = sim
        
        # Also match against ontology text embeddings
        for cls, emb in self.onto_text_embs.items():
            onto_sim = np.dot(txt_vec, emb)
            scores[cls] = max(scores.get(cls, 0), onto_sim * 0.8)  # blend
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def update(self, cls, resnet_feat, clip_feat):
        tpl = self.templates[cls]
        tpl["resnet_features"].append(resnet_feat)
        tpl["clip_vis_features"].append(clip_feat)
        n = len(tpl["resnet_features"])
        tpl["resnet_centroid"] = tpl["resnet_centroid"] * (n-1)/n + resnet_feat/n
        tpl["clip_vis_centroid"] = tpl["clip_vis_centroid"] * (n-1)/n + clip_feat/n
        tpl["n"] = n


def main():
    print("=" * 64)
    print("双索引增强KG原型 — ResNet18 + CN-CLIP 跨模态检索")
    print("=" * 64)
    
    # ---- Init ----
    print("\n[1] 初始化双索引 (ResNet18 + CN-CLIP)...")
    kg = DualIndex()
    print("    ResNet18: ✓ | CN-CLIP: ✓")
    
    # ---- Load data ----
    print("\n[2] 加载 DTD 数据...")
    class_images = {}
    for cls in FABRIC_CLASSES:
        cls_dir = DTD_DIR / cls
        if not cls_dir.exists():
            continue
        imgs = sorted(cls_dir.glob("*.jpg"))[:SEED_PER_CLASS + TEST_PER_CLASS]
        if len(imgs) >= SEED_PER_CLASS + 5:
            class_images[cls] = imgs
    print(f"    {len(class_images)} 类")
    
    # ---- Cold-start ----
    print(f"\n[3] 冷启动 (每类 {SEED_PER_CLASS} 种子)...")
    test_sets = {}
    for cls, imgs in class_images.items():
        kg.seed(cls, imgs[:SEED_PER_CLASS])
        test_sets[cls] = imgs[SEED_PER_CLASS:SEED_PER_CLASS + TEST_PER_CLASS]
    
    # Baseline: ResNet visual retrieval
    correct_r = 0; total = 0
    for cls, imgs in test_sets.items():
        for img in imgs:
            feat = kg.extract_resnet(img)
            results = kg.query_visual(feat, top_k=1, use_clip=False)
            if results and results[0][0] == cls:
                correct_r += 1
            total += 1
    acc_resnet = correct_r / total
    print(f"    ResNet 冷启动: {correct_r}/{total} = {acc_resnet:.1%}")
    
    # Baseline: CLIP visual retrieval
    correct_c = 0; total = 0
    for cls, imgs in test_sets.items():
        for img in imgs:
            feat = kg.extract_clip_vis(img)
            results = kg.query_visual(feat, top_k=1, use_clip=True)
            if results and results[0][0] == cls:
                correct_c += 1
            total += 1
    acc_clip = correct_c / total
    print(f"    CLIP 冷启动:  {correct_c}/{total} = {acc_clip:.1%}")
    
    # ---- Text-based search demo ----
    print(f"\n[4] 文本→图像 跨模态检索演示")
    queries = [
        "规则的格子图案，类似棋盘格",
        "弯曲的泪滴形花纹，华丽的涡卷图案",
        "平行排列的条纹",
        "V形环状的针织纹理",
        "圆形斑点均匀分布",
    ]
    for q in queries:
        results = kg.query_text(q, top_k=3)
        top = results[0]
        onto = PATTERN_ONTOLOGY.get(top[0], {})
        print(f"  查询: \"{q}\"")
        print(f"  → Top-1: {top[0]} ({top[1]:.3f}) [{onto.get('type','?')}]")
        print(f"    Top-3: {', '.join(f'{c}({s:.3f})' for c,s in results)}")
    
    # ---- KG ontology text embeddings ----
    print(f"\n[5] KG 本体描述嵌入 (CN-CLIP 文本编码)")
    for cls in ["woven", "knitted", "paisley", "dotted"]:
        onto = PATTERN_ONTOLOGY[cls]
        desc = onto["visual"]
        print(f"   [{cls}]: \"{desc[:50]}...\"")

    # ---- Summary ----
    print(f"\n{'='*64}")
    print(f"[总结]")
    print(f"  ResNet 视觉检索: {acc_resnet:.1%}")
    print(f"  CN-CLIP 视觉检索: {acc_clip:.1%}")
    print(f"  CN-CLIP 文本检索: ✓ 可用 (跨模态)")
    print(f"  双索引架构: 视觉相似 + 语义对齐 — 完整的多模态增强索引")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
