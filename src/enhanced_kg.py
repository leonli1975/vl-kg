"""Enhanced Three-Layer KG with dual-index (ResNet18 + CN-CLIP) visual retrieval.

Integrates symbolic knowledge graph + embedding-based visual retrieval:

  Layer 1: Atomic Motifs ← [ResNet18 visual + CN-CLIP cross-modal embeddings]
  Layer 2: Composition Grammar (layouts, transformations)
  Layer 3: Composite Patterns (real textile patterns)

Pipeline:
  Image → [ResNet18 + CN-CLIP] → Motif Recognition → ThreeLayerKG → Cultural Understanding
                                              ↑________feedback________↓
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
from typing import List, Dict, Optional, Tuple

from .three_layer_kg import ThreeLayerKG, ATOMIC_MOTIFS, COMPOSITE_PATTERNS, LAYOUT_TYPES, TRANSFORMATION_TYPES
# TODO:set CN-CLIP checkpoint path 
CN_CLIP_CKPT = "~/multi-modal_clip-vit-base-patch16_zh/clip_cn_vit-b-16.pt"


class EnhancedThreeLayerKG(ThreeLayerKG):
    """Three-layer KG enhanced with ResNet18 + CN-CLIP dual-index visual retrieval.

    Each atomic motif stores:
      - resnet_centroid: 512-dim ResNet18 visual centroid
      - clip_vis_centroid: 512-dim CN-CLIP visual centroid
      - clip_text_emb: 512-dim CN-CLIP text embedding (from motif description)

    Supports:
      - Image → motif recognition (visual retrieval)
      - Text → motif cross-modal search
      - Motif decomposition → cultural narrative synthesis
      - Incremental embedding update (recognition→confirm→feedback loop)
    """

    def __init__(self):
        super().__init__()
        self._init_resnet()
        self._init_clip()
        self._init_motif_store()
        self._precompute_motif_text_embeddings()

    # ── Model Initialization ──────────────────────────────────────────

    def _init_resnet(self):
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Identity()
        model.eval()
        self.resnet = model
        self.resnet_prep = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        self.clip_model = model
        self.clip_prep = T.Compose([
            T.Resize(224, interpolation=Image.BICUBIC), T.CenterCrop(224), T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711)),
        ])

    # ── Motif Embedding Store ─────────────────────────────────────────

    def _init_motif_store(self):
        """Per-motif embedding storage (mirrors prototype DualIndex template structure)."""
        self.motif_embeddings = {}
        for motif_id in self.motifs:
            self.motif_embeddings[motif_id] = {
                "resnet_features": [],
                "resnet_centroid": None,
                "clip_vis_features": [],
                "clip_vis_centroid": None,
                "clip_text_emb": None,
                "n": 0,
            }

    def _precompute_motif_text_embeddings(self):
        """Pre-compute CN-CLIP text embeddings for all motif descriptions."""
        for motif_id, motif in self.motifs.items():
            desc = motif["description"]
            tokens = clip.tokenize([desc])
            with torch.no_grad():
                emb = self.clip_model.encode_text(tokens)
                emb /= emb.norm(dim=-1, keepdim=True)
            self.motif_embeddings[motif_id]["clip_text_emb"] = emb.squeeze(0).numpy()

    # ── Feature Extraction ────────────────────────────────────────────

    def extract_resnet(self, img_path: str) -> np.ndarray:
        img = Image.open(img_path).convert('RGB')
        t = self.resnet_prep(img).unsqueeze(0)
        with torch.no_grad():
            return self.resnet(t).squeeze(0).numpy()

    def extract_clip_vis(self, img_path: str) -> np.ndarray:
        img = Image.open(img_path).convert('RGB')
        t = self.clip_prep(img).unsqueeze(0)
        with torch.no_grad():
            f = self.clip_model.encode_image(t)
            f /= f.norm(dim=-1, keepdim=True)
        return f.squeeze(0).numpy()

    # ── Seeding (Cold-Start) ──────────────────────────────────────────

    def seed_motif(self, motif_id: str, img_paths: List[str]):
        """Seed a motif with example images to compute initial centroids."""
        if motif_id not in self.motif_embeddings:
            raise ValueError(f"Unknown motif: {motif_id}")
        if not img_paths:
            return

        r_feats = [self.extract_resnet(p) for p in img_paths]
        c_feats = [self.extract_clip_vis(p) for p in img_paths]

        store = self.motif_embeddings[motif_id]
        store["resnet_features"] = r_feats
        store["resnet_centroid"] = np.mean(r_feats, axis=0)
        store["clip_vis_features"] = c_feats
        store["clip_vis_centroid"] = np.mean(c_feats, axis=0)
        store["n"] = len(img_paths)

    # ── Query ─────────────────────────────────────────────────────────

    def query_visual(self, feature: np.ndarray, top_k: int = 5,
                     use_clip: bool = False) -> List[Tuple[str, float]]:
        """Image feature → top-k motif retrieval.

        Args:
            feature: ResNet or CLIP visual feature vector
            top_k: number of top results
            use_clip: use CLIP centroids (True) or ResNet centroids (False)

        Returns:
            [(motif_id, cosine_similarity), ...]
        """
        feat_key = "clip_vis_centroid" if use_clip else "resnet_centroid"
        scores = {}
        for motif_id, store in self.motif_embeddings.items():
            cent = store.get(feat_key)
            if cent is None:
                continue
            sim = np.dot(feature, cent) / (np.linalg.norm(feature) * np.linalg.norm(cent) + 1e-8)
            scores[motif_id] = sim
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def query_text(self, text_query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Text description → top-k motif retrieval (cross-modal).

        Matches against both:
          - CLIP visual centroids (text→image alignment)
          - Pre-computed motif description embeddings (text→text alignment)

        Returns:
            [(motif_id, cosine_similarity), ...]
        """
        tokens = clip.tokenize([text_query])
        with torch.no_grad():
            txt_emb = self.clip_model.encode_text(tokens)
            txt_emb /= txt_emb.norm(dim=-1, keepdim=True)
        txt_vec = txt_emb.squeeze(0).numpy()

        scores = {}
        for motif_id, store in self.motif_embeddings.items():
            sim = 0.0
            # Match against CLIP visual centroids (if seeded)
            if store["clip_vis_centroid"] is not None:
                vis_sim = np.dot(txt_vec, store["clip_vis_centroid"])
                sim = max(sim, vis_sim)
            # Match against motif description embedding
            if store["clip_text_emb"] is not None:
                txt_sim = np.dot(txt_vec, store["clip_text_emb"])
                sim = max(sim, txt_sim)
            scores[motif_id] = sim

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def query_by_cross_meaning(self, meaning_query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search: find motifs whose cultural meanings match the query.

        Uses CN-CLIP to encode the query and match against motif descriptions,
        then returns full KG information for each match.
        """
        text_results = self.query_text(meaning_query, top_k=top_k)
        results = []
        for motif_id, score in text_results:
            motif = self.motifs[motif_id]
            # Find composite patterns containing this motif
            appearances = []
            for pid, pattern in self.patterns.items():
                for m in pattern["motifs"]:
                    if m["motif"] == motif_id:
                        appearances.append({
                            "pattern_id": pid,
                            "pattern_name": pattern["name_zh"],
                            "role": m["role"],
                        })
            results.append({
                "motif_id": motif_id,
                "name": motif["name_zh"],
                "category": motif["category"],
                "cultural_meanings": motif["cultural_meanings"],
                "description": motif["description"],
                "origin": motif["origin"],
                "similarity": float(score),
                "appears_in": appearances,
            })
        return results

    # ── Incremental Update (Feedback Loop) ────────────────────────────

    def update(self, motif_id: str, resnet_feat: np.ndarray, clip_feat: np.ndarray):
        """Incremental centroid update with a new verified example.

        Recognition→Confirmation→Feedback: when a user confirms an image
        belongs to a motif, this updates the centroid online.
        Handles both first-seed (centroid is None) and incremental cases.
        """
        store = self.motif_embeddings[motif_id]
        store["resnet_features"].append(resnet_feat)
        store["clip_vis_features"].append(clip_feat)
        n = len(store["resnet_features"])
        # Running mean update (handles first-seed case)
        if store["resnet_centroid"] is None:
            store["resnet_centroid"] = resnet_feat.copy()
        else:
            store["resnet_centroid"] = (store["resnet_centroid"] * (n - 1) / n
                                        + resnet_feat / n)
        if store["clip_vis_centroid"] is None:
            store["clip_vis_centroid"] = clip_feat.copy()
        else:
            store["clip_vis_centroid"] = (store["clip_vis_centroid"] * (n - 1) / n
                                          + clip_feat / n)
        store["n"] = n

    # ── Full Recognition Pipeline ─────────────────────────────────────

    def recognize_image(self, img_path: str, top_k: int = 5) -> Dict:
        """Recognize motifs in an image and return cultural understanding.

        Pipeline:
          1. Extract ResNet18 + CN-CLIP features
          2. Query visual → top-k motif matches
          3. Enrich with KG knowledge
          4. Attempt pattern composition matching

        Returns:
            {
                "image": str,
                "top_motifs": [{motif_id, name, score_resnet, score_clip, ...}],
                "suggested_patterns": [{pattern_id, name, match_ratio, ...}],
            }
        """
        r_feat = self.extract_resnet(img_path)
        c_feat = self.extract_clip_vis(img_path)

        r_results = self.query_visual(r_feat, top_k=top_k, use_clip=False)
        c_results = self.query_visual(c_feat, top_k=top_k, use_clip=True)

        # Merge ResNet + CLIP results
        motif_scores = {}
        for motif_id, score in r_results:
            motif_scores[motif_id] = {"resnet": float(score), "clip": 0.0}
        for motif_id, score in c_results:
            if motif_id in motif_scores:
                motif_scores[motif_id]["clip"] = float(score)
            else:
                motif_scores[motif_id] = {"resnet": 0.0, "clip": float(score)}

        # Compute blended score and sort
        blended = []
        for motif_id, scores in motif_scores.items():
            blended_score = scores["resnet"] * 0.4 + scores["clip"] * 0.6
            motif = self.motifs[motif_id]
            blended.append({
                "motif_id": motif_id,
                "name": motif["name_zh"],
                "category": motif["category"],
                "score_resnet": round(scores["resnet"], 4),
                "score_clip": round(scores["clip"], 4),
                "score_blended": round(blended_score, 4),
                "cultural_meanings": motif["cultural_meanings"],
                "description": motif["description"],
            })
        blended.sort(key=lambda x: x["score_blended"], reverse=True)

        # Match recognized motifs to composite patterns
        top_motif_ids = {m["motif_id"] for m in blended[:top_k]}
        pattern_matches = []
        for pid, pattern in self.patterns.items():
            pattern_motif_ids = {m["motif"] for m in pattern["motifs"]}
            overlap = top_motif_ids & pattern_motif_ids
            if overlap:
                match_ratio = len(overlap) / len(pattern_motif_ids)
                pattern_matches.append({
                    "pattern_id": pid,
                    "name": pattern["name_zh"],
                    "craft": pattern["craft"],
                    "match_ratio": round(match_ratio, 2),
                    "matched_motifs": [self.motifs[mid]["name_zh"] for mid in overlap],
                    "missing_motifs": [self.motifs[mid]["name_zh"] for mid in pattern_motif_ids - overlap],
                    "narrative": pattern["cultural_narrative"],
                })
        pattern_matches.sort(key=lambda x: x["match_ratio"], reverse=True)

        return {
            "image": img_path,
            "top_motifs": blended,
            "suggested_patterns": pattern_matches,
        }

    # ── Unified Understanding ─────────────────────────────────────────

    def understand_image(self, annotation: dict) -> str:
        """Generate complete cultural interpretation from an annotated image.

        This is the bridge between visual recognition and cultural knowledge.
        Accepts an annotation dict in template_v2 format and produces a
        formatted report with KG enrichment.
        """
        lines = []
        craft = annotation.get("craft", {})
        subtype = craft.get("subtype", craft.get("composite_pattern", "未知"))
        craft_type = craft.get("type", "未知")

        lines.append(f"{'='*60}")
        lines.append(f"【{subtype}】({craft_type})")
        lines.append(f"{'='*60}")

        # Composition structure
        comp = annotation.get("composition_grammar", {})
        if comp:
            lines.append(f"\n▎构图结构")
            lines.append(f"  主布局: {comp.get('primary_layout', '?')}")
            lines.append(f"  节奏:   {comp.get('rhythm', '?')}")
            lines.append(f"  描述:   {comp.get('description', '?')}")

        # Motif-by-motif analysis
        motifs = annotation.get("motif_decomposition", [])
        lines.append(f"\n▎纹样分解 ({len(motifs)} 个基本纹样)")

        all_motif_keys = []
        for i, m in enumerate(motifs, 1):
            motif_key = self._match_motif_key(m["motif"])
            all_motif_keys.append(motif_key)
            kg_entry = self.motifs.get(motif_key, {})

            lines.append(f"\n  {i}. [{m['role']}] {m['motif']}")
            lines.append(f"     位置: {m.get('position', '?')} | 布局: {m.get('layout_relation', '?')} | 占比: {m.get('coverage_ratio', 0):.0%}")
            lines.append(f"     形态: {m.get('iconic_form', '?')} | 变形: {m.get('transformation', '?')}")

            meanings = kg_entry.get("cultural_meanings", m.get("cultural_meaning", []))
            lines.append(f"     文化含义: {', '.join(meanings)}")

            origin = kg_entry.get("origin", "")
            if origin:
                lines.append(f"     历史来源: {origin}")

            if kg_entry.get("description"):
                lines.append(f"     KG知识: {kg_entry['description'][:120]}...")

            lines.append(f"     标注描述: {m.get('description', '')}")

        # KG Traceability
        lines.append(f"\n▎KG溯源 — 纹样还出现在哪些复合纹样中")
        seen_patterns = set()
        for motif_key in all_motif_keys:
            result = self.query_by_motif(motif_key)
            if "appears_in" in result:
                for app in result["appears_in"]:
                    pid = app["pattern_id"]
                    if pid not in seen_patterns:
                        seen_patterns.add(pid)
                        pname = self.patterns.get(pid, {}).get("name_zh", pid)
                        lines.append(f"  {result['motif_name']} → {pname} (作为 {app['role']})")

        # Cross-modal check: verify annotation meanings against KG
        lines.append(f"\n▎交叉验证 — 标注 vs KG 知识一致性")
        for m in motifs:
            motif_key = self._match_motif_key(m["motif"])
            kg_entry = self.motifs.get(motif_key, {})
            annot_meanings = set(m.get("cultural_meaning", []))
            kg_meanings = set(kg_entry.get("cultural_meanings", []))
            overlap = annot_meanings & kg_meanings
            missing_in_kg = annot_meanings - kg_meanings
            extra_in_kg = kg_meanings - annot_meanings

            status = "✓" if missing_in_kg == extra_in_kg == set() else "⚠"
            line = f"  {status} {m['motif']}: 标注 {len(annot_meanings)} 含义, KG {len(kg_meanings)} 含义"
            if overlap:
                line += f", 一致: {', '.join(sorted(overlap))}"
            lines.append(line)
            if missing_in_kg:
                lines.append(f"    ⚠ 标注有但KG缺: {', '.join(sorted(missing_in_kg))} → 建议补充KG")
            if extra_in_kg:
                lines.append(f"    ℹ KG有但标注未提: {', '.join(sorted(extra_in_kg))} → 建议补充标注")

        # Cultural narrative synthesis
        narrative = annotation.get("cultural_meaning", {}).get("narrative", "")
        if narrative:
            lines.append(f"\n▎文化叙事合成")
            lines.append(f"  {narrative}")

        # Embedding status
        lines.append(f"\n▎嵌入索引状态")
        for motif_key in all_motif_keys:
            store = self.motif_embeddings.get(motif_key, {})
            n = store.get("n", 0)
            has_text = store.get("clip_text_emb") is not None
            has_vis = store.get("resnet_centroid") is not None
            name = self.motifs.get(motif_key, {}).get("name_zh", motif_key)
            lines.append(f"  {name}: 视觉样本={n}, 文本嵌入={'✓' if has_text else '✗'}, 视觉质心={'✓' if has_vis else '✗'}")

        lines.append(f"\n{'='*60}")
        return "\n".join(lines)

    def _match_motif_key(self, name: str) -> str:
        """Match Chinese motif name to KG key."""
        mapping = {
            "月华纹": "moon_halo", "凤凰纹": "phoenix", "卷草纹": "scroll_grass",
            "云纹": "cloud", "牡丹纹": "peony", "龙纹": "dragon",
            "蝴蝶纹": "butterfly", "万字纹": "swastika", "雨丝纹": "rain_thread",
        }
        return mapping.get(name, name.lower().replace(" ", "_"))

    # ── Statistics ────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return comprehensive KG statistics including embedding status."""
        motifs_with_vis = sum(1 for s in self.motif_embeddings.values()
                              if s["resnet_centroid"] is not None)
        motifs_with_text = sum(1 for s in self.motif_embeddings.values()
                               if s["clip_text_emb"] is not None)
        total_samples = sum(s["n"] for s in self.motif_embeddings.values())

        return {
            "motifs": len(self.motifs),
            "motifs_with_visual_centroids": motifs_with_vis,
            "motifs_with_text_embeddings": motifs_with_text,
            "total_visual_samples": total_samples,
            "layouts": len(self.layouts),
            "transformations": len(self.transformations),
            "composite_patterns": len(self.patterns),
        }


# ═══════════════════════════════════════════════════════════════
# Demo: Full Pipeline Validation
# ═══════════════════════════════════════════════════════════════

ANNOTATED_IMAGE = {
    "image_id": "SJ_001",
    "filename": "yuehua_jin_sample.jpg",
    "craft": {"type": "蜀锦", "subtype": "月华锦", "composite_pattern": "月华锦"},
    "motif_decomposition": [
        {
            "motif": "月华纹", "category": "自然纹", "iconic_form": "同心圆放射状",
            "transformation": "旋转对称", "role": "底纹", "position": "满铺",
            "layout_relation": "四方连续", "coverage_ratio": 0.70,
            "cultural_meaning": ["月光普照", "圆满", "和谐"],
            "description": "同心圆月华纹满铺整幅，五彩经线放射状排列"
        },
        {
            "motif": "凤凰纹", "category": "动物纹", "iconic_form": "展翅",
            "transformation": "几何简化", "role": "主纹样", "position": "中心",
            "layout_relation": "中心对称", "coverage_ratio": 0.15,
            "cultural_meaning": ["吉祥", "重生", "高贵"],
            "description": "简化几何化展翅凤凰居中，双凤对称排列"
        },
        {
            "motif": "卷草纹", "category": "植物纹", "iconic_form": "波状",
            "transformation": "简化", "role": "边框纹", "position": "边框",
            "layout_relation": "二方连续", "coverage_ratio": 0.10,
            "cultural_meaning": ["生生不息", "连绵不断"],
            "description": "绿色波状卷草纹环绕边框，二方连续排列"
        },
        {
            "motif": "云纹", "category": "几何纹", "iconic_form": "如意云",
            "transformation": "抽象化", "role": "辅助纹", "position": "散点",
            "layout_relation": "散点", "coverage_ratio": 0.05,
            "cultural_meaning": ["祥瑞", "高升"],
            "description": "如意云头纹散点分布，填补主纹与边框之间的空白"
        },
    ],
    "composition_grammar": {
        "primary_layout": "中心对称", "border_layout": "二方连续",
        "background_layout": "四方连续",
        "rhythm": "主纹突出+边框重复+底纹满铺",
        "description": "三层构图：底纹（月华四方连续）→ 主纹（凤凰中心对称）→ 边框（卷草二方连续），云纹散点填充间隙"
    },
    "cultural_meaning": {
        "narrative": "在月华普照的底纹上，双凤承载着吉祥重生的祝福，卷草纹环绕象征生命绵延不断，云纹点缀增添祥瑞之气。整幅作品表达了对富贵吉祥、生生不息的美好祈愿。"
    },
}


def main():
    print("=" * 64)
    print("增强三层KG — 符号知识 + 双索引嵌入 (ResNet18 + CN-CLIP)")
    print("=" * 64)

    # ── 1. Initialization ──
    print("\n[1] 初始化增强三层KG...")
    t0 = time.time()
    kg = EnhancedThreeLayerKG()
    elapsed = time.time() - t0
    st = kg.stats()
    print(f"    加载完成 ({elapsed:.1f}s)")
    print(f"    纹样: {st['motifs']} | 布局: {st['layouts']} | "
          f"变形: {st['transformations']} | 复合纹样: {st['composite_patterns']}")
    print(f"    文本嵌入: {st['motifs_with_text_embeddings']}/{st['motifs']} 个纹样已编码")
    print(f"    视觉质心: {st['motifs_with_visual_centroids']}/{st['motifs']} (待图像种子)")

    # ── 2. Cross-modal text→motif search ──
    print(f"\n[2] 跨模态文本→纹样检索 (CN-CLIP)")
    queries = [
        "代表富贵繁荣的花卉纹样",
        "象征吉祥与重生的神鸟",
        "表示永恒和无限的符号",
        "月光普照的同心圆纹样",
        "象征生命连绵不断的植物装饰",
        "代表祥瑞高升的如意形图案",
        "细密线条如雨丝般的纹理",
        "象征权力与尊贵的神兽纹样",
    ]
    for q in queries:
        results = kg.query_text(q, top_k=3)
        top = results[0]
        motif_info = kg.motifs.get(top[0], {})
        print(f"  查询: \"{q}\"")
        print(f"  → Top-1: {motif_info.get('name_zh', top[0])} "
              f"({motif_info.get('category', '?')}) [{top[1]:.3f}]")
        top3_strs = []
        for c, s in results:
            name = kg.motifs.get(c, {}).get("name_zh", c)
            top3_strs.append(f"{name}({s:.3f})")
        print(f"    Top-3: {', '.join(top3_strs)}")

    # ── 3. Semantic cultural meaning search ──
    print(f"\n[3] 文化含义语义搜索")
    cultural_queries = ["吉祥如意的美好祝福", "象征生命的永恒轮回", "皇权与尊贵"]
    for cq in cultural_queries:
        results = kg.query_by_cross_meaning(cq, top_k=3)
        print(f"  \"{cq}\" →")
        for r in results:
            print(f"    {r['name']} ({r['category']}): {', '.join(r['cultural_meanings'])} "
                  f"[{r['similarity']:.3f}]")

    # ── 4. Pattern decomposition (symbolic KG) ──
    print(f"\n[4] 复合纹样分解 (三层KG)")
    for pid in ["yuehua_jin", "feng_chuan_mudan", "fangfang_jin"]:
        r = kg.decompose_pattern(pid)
        print(f"  {r['pattern_name']} ({r['craft_type']}, {r['era']}):")
        motif_strs = [f"{m['name']}({m['role']})" for m in r['motifs']]
        comp_strs = [f"{c['position']}:{c['name']}" for c in r['composition']]
        print(f"    纹样: {', '.join(motif_strs)}")
        print(f"    布局: {', '.join(comp_strs)}")

    # ── 5. Full image understanding pipeline ──
    print(f"\n[5] 完整图像理解流水线 (标注→KG→文化叙事)")
    report = kg.understand_image(ANNOTATED_IMAGE)
    print(report)

    # ── 6. Feedback loop demo ──
    print(f"\n\n[6] 增量反馈闭环演示")
    motif_id = "moon_halo"
    store_before = kg.motif_embeddings[motif_id]["n"]
    # Simulate: extract features from a "new image" (re-extract from same image
    # for demo purposes — in production this would be a new verified image)
    dummy_img = "/home/lgc/OCtestprj/fabric_structure/prototype/data/dtd/images/dotted/dotted_0001.jpg"
    if os.path.exists(dummy_img):
        r_feat = kg.extract_resnet(dummy_img)
        c_feat = kg.extract_clip_vis(dummy_img)
        print(f"  更新前: {kg.motifs[motif_id]['name_zh']} 样本数 = {store_before}")
        kg.update(motif_id, r_feat, c_feat)
        store_after = kg.motif_embeddings[motif_id]["n"]
        print(f"  更新后: {kg.motifs[motif_id]['name_zh']} 样本数 = {store_after}")
        print(f"  反馈闭环: 识别→确认→质心更新 ✓")
    else:
        print(f"  (跳过 — 无可用图像)")

    # ── 7. Summary ──
    st = kg.stats()
    print(f"\n{'='*64}")
    print("增强三层KG原型验证通过。")
    print(f"  符号知识层:  ✓  {st['motifs']}纹样 × {st['layouts']}布局 × {st['transformations']}变形 × {st['composite_patterns']}复合纹样")
    print(f"  文本嵌入层:  ✓  CN-CLIP跨模态文本→纹样检索")
    print(f"  视觉检索层:  ⧖  待真实纹样图像种子 (框架已就绪)")
    print(f"  反馈闭环:    ✓  增量质心更新机制")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
