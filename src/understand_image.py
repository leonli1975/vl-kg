"""Motif decomposition + KG understanding pipeline (powered by EnhancedThreeLayerKG).

Input: annotated image → Output: complete cultural interpretation
Demonstrates: 基本纹样分解 → KG查询 → 文化叙事合成 → 跨模态语义检索
"""

import json
from pathlib import Path
from .enhanced_kg import EnhancedThreeLayerKG
from .three_layer_kg import ATOMIC_MOTIFS, COMPOSITE_PATTERNS


# ── Annotated image example (using template_v2 format) ──

ANNOTATED_IMAGE = {
    "image_id": "SJ_001",
    "filename": "yuehua_jin_sample.jpg",
    "craft": {"type": "蜀锦", "subtype": "月华锦", "composite_pattern": "月华锦"},

    "motif_decomposition": [
        {
            "motif": "月华纹",
            "category": "自然纹",
            "iconic_form": "同心圆放射状",
            "transformation": "旋转对称",
            "role": "底纹",
            "position": "满铺",
            "layout_relation": "四方连续",
            "coverage_ratio": 0.70,
            "cultural_meaning": ["月光普照", "圆满", "和谐"],
            "description": "同心圆月华纹满铺整幅，五彩经线放射状排列，底纹占据画面70%"
        },
        {
            "motif": "凤凰纹",
            "category": "动物纹",
            "iconic_form": "展翅",
            "transformation": "几何简化",
            "role": "主纹样",
            "position": "中心",
            "layout_relation": "中心对称",
            "coverage_ratio": 0.15,
            "cultural_meaning": ["吉祥", "重生", "高贵"],
            "description": "简化几何化展翅凤凰居中，双凤对称排列"
        },
        {
            "motif": "卷草纹",
            "category": "植物纹",
            "iconic_form": "波状",
            "transformation": "简化",
            "role": "边框纹",
            "position": "边框",
            "layout_relation": "二方连续",
            "coverage_ratio": 0.10,
            "cultural_meaning": ["生生不息", "连绵不断"],
            "description": "绿色波状卷草纹环绕边框，二方连续排列"
        },
        {
            "motif": "云纹",
            "category": "几何纹",
            "iconic_form": "如意云",
            "transformation": "抽象化",
            "role": "辅助纹",
            "position": "散点",
            "layout_relation": "散点",
            "coverage_ratio": 0.05,
            "cultural_meaning": ["祥瑞", "高升"],
            "description": "如意云头纹散点分布，填补主纹与边框之间的空白"
        },
    ],

    "composition_grammar": {
        "primary_layout": "中心对称",
        "border_layout": "二方连续",
        "background_layout": "四方连续",
        "rhythm": "主纹突出+边框重复+底纹满铺",
        "description": "三层构图：底纹（月华四方连续）→ 主纹（凤凰中心对称）→ 边框（卷草二方连续），云纹散点填充间隙"
    },

    "cultural_meaning": {
        "narrative": "在月华普照的底纹上，双凤承载着吉祥重生的祝福，卷草纹环绕象征生命绵延不断，云纹点缀增添祥瑞之气。整幅作品表达了对富贵吉祥、生生不息的美好祈愿。"
    },
}


# ── Understanding Pipeline ──

class ImageUnderstandingPipeline:
    """Decompose image → query KG → synthesize cultural understanding.
    
    Powered by EnhancedThreeLayerKG for both symbolic and embedding-based retrieval.
    """

    def __init__(self, kg: EnhancedThreeLayerKG):
        self.kg = kg

    def decompose(self, annotation: dict) -> dict:
        """Decompose annotated image into motifs with KG enrichment."""
        motifs = []
        for m in annotation["motif_decomposition"]:
            motif_key = self._match_motif_key(m["motif"])
            kg_entry = self.kg.motifs.get(motif_key, {})
            motifs.append({
                "name": m["motif"],
                "role": m["role"],
                "position": m["position"],
                "layout": m["layout_relation"],
                "coverage": f"{m['coverage_ratio']:.0%}",
                "transformation": m.get("transformation", ""),
                "cultural_meanings": kg_entry.get("cultural_meanings", m.get("cultural_meaning", [])),
                "origin": kg_entry.get("origin", "未知"),
                "iconic_forms": kg_entry.get("iconic_forms", []),
                "description": m.get("description", ""),
            })
        return {
            "image_id": annotation["image_id"],
            "craft": annotation["craft"],
            "motifs": motifs,
            "composition": annotation["composition_grammar"],
        }

    def understand(self, annotation: dict) -> str:
        """Generate complete cultural understanding of an image.
        
        Uses EnhancedThreeLayerKG.understand_image() for the full pipeline,
        which adds embedding status and cross-validation checks.
        """
        return self.kg.understand_image(annotation)

    def cross_modal_search(self, text_query: str, top_k: int = 3) -> list:
        """Search for motifs by semantic description (CN-CLIP cross-modal)."""
        return self.kg.query_by_cross_meaning(text_query, top_k=top_k)

    def _match_motif_key(self, name: str) -> str:
        """Match Chinese motif name to KG key."""
        mapping = {
            "月华纹": "moon_halo", "凤凰纹": "phoenix", "卷草纹": "scroll_grass",
            "云纹": "cloud", "牡丹纹": "peony", "龙纹": "dragon",
            "蝴蝶纹": "butterfly", "万字纹": "swastika", "雨丝纹": "rain_thread",
        }
        return mapping.get(name, name.lower().replace(" ", "_"))


def main():
    kg = EnhancedThreeLayerKG()
    pipeline = ImageUnderstandingPipeline(kg)

    print("=" * 60)
    print("图像理解流程: 纹样分解 → KG查询 → 文化理解")
    print("(基于 EnhancedThreeLayerKG: 符号KG + 双索引嵌入)")
    print("=" * 60)

    # Demo 1: Full understanding of annotated image (now uses enhanced KG)
    understanding = pipeline.understand(ANNOTATED_IMAGE)
    print(understanding)

    # Demo 2: Cross-modal semantic search
    print(f"\n{'='*60}")
    print("跨模态语义检索: 自然语言 → 纹样匹配")
    print("=" * 60)
    semantic_queries = [
        "代表吉祥与重生的神鸟纹样",
        "象征月光普照和圆满的底纹",
        "表示生生不息和连绵不断的边框装饰",
        "富贵繁荣的花卉纹样",
    ]
    for q in semantic_queries:
        results = pipeline.cross_modal_search(q, top_k=3)
        print(f"\n  查询: \"{q}\"")
        for r in results:
            print(f"    → {r['name']} ({r['category']}) [{r['similarity']:.3f}]")
            print(f"       含义: {', '.join(r['cultural_meanings'])}")
            if r['appears_in']:
                patterns = [a['pattern_name'] for a in r['appears_in']]
                print(f"       出现在: {', '.join(patterns)}")

    # Demo 3: Cross-reference check (annotation vs KG)
    print(f"\n{'='*60}")
    print("交叉验证: 标注 vs KG 知识一致性检查")
    print("=" * 60)
    for m in ANNOTATED_IMAGE["motif_decomposition"]:
        key = pipeline._match_motif_key(m["motif"])
        kg_entry = kg.motifs.get(key, {})
        annot_meanings = set(m.get("cultural_meaning", []))
        kg_meanings = set(kg_entry.get("cultural_meanings", []))
        overlap = annot_meanings & kg_meanings
        missing_in_kg = annot_meanings - kg_meanings
        extra_in_kg = kg_meanings - annot_meanings
        print(f"\n  {m['motif']}:")
        print(f"    标注含义: {annot_meanings}")
        print(f"    KG含义:   {kg_meanings}")
        print(f"    一致:     {overlap if overlap else '(无)'}")
        if missing_in_kg:
            print(f"    ⚠ 标注有但KG缺: {missing_in_kg} → 建议补充KG")
        if extra_in_kg:
            print(f"    ℹ KG有但标注未提: {extra_in_kg} → 建议补充标注")

    # Demo 4: KG statistics
    print(f"\n{'='*60}")
    print("KG统计")
    print("=" * 60)
    st = kg.stats()
    for k, v in st.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*60}")
    print("验证完成。增强KG已就绪。")
    print("=" * 60)


if __name__ == "__main__":
    main()
