"""Qwen2.5-VL zero-shot motif recognition via DashScope API.

Pipeline:
  Image → Qwen2.5-VL → Structured Motif List → KG Enrichment → Cultural Narrative

Three recognition tasks:
  1. motif_decomposition:  identify motifs, roles, positions, coverage
  2. motif_detail:         cultural meaning, origin, iconic form
  3. composition:          layout type, rhythm, grammar analysis
"""

import base64
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from openai import OpenAI
from .three_layer_kg import ATOMIC_MOTIFS, COMPOSITE_PATTERNS, LAYOUT_TYPES

# ── Config ──
QWEN_VL_CONFIG = {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "sk-ws-H.RELYHIL.qc5x.MEUCIQCow3oV19Z0UBJNbRdZeWq8FbQVjBugeC20N_IPLU_3FQIgbQuWA_PL4TXNEjlQ9t08HFybgnLBpjs6K-oAUr9GDUM",
    "model": "qwen-vl-plus",
}

# ═══════════════════════════════════════════════════════════════
# Prompt Templates
# ═══════════════════════════════════════════════════════════════

MOTIF_TAXONOMY = """蜀锦/蜀绣纹样分类体系：
动物纹：凤凰纹、龙纹、蝴蝶纹、麒麟纹、仙鹤纹、鸳鸯纹、孔雀纹、蝙蝠纹、鹿纹、狮纹、鱼纹
植物纹：卷草纹、牡丹纹、莲花纹、宝相花纹、梅花纹、茱萸纹
几何纹：云纹、万字纹、回纹、龟背纹、联珠纹
文字纹：福/寿/喜/贵等吉祥文字纹"""

MOTIF_DECOMPOSITION_PROMPT = """你是一位蜀锦/蜀绣纹样鉴定专家。请仔细分析这幅织物图像，识别其中的纹样元素。

""" + MOTIF_TAXONOMY + """

参考信息——蜀锦经典复合纹样及其典型纹样组合：
- 月华锦（彩条晕涧结构）：{凤凰, 卷草, 云纹} 寓意月光普照，富贵生生不息
- 雨丝锦（雨丝结构）：{龙, 凤凰, 牡丹} 经丝彩色渐变形成雨条效果
- 方方锦（方格结构）：{牡丹, 万字, 梅花} 经纬彩格骨架内饰圆形花纹
- 凤穿牡丹：{凤凰, 牡丹} 凤为主体，牡丹环绕，富贵吉祥
- 万字锦：{万字(底纹), 散点小花} 卍字四方连续
- 铺地锦：{几何地纹, 牡丹(主花), 卷草(边框)} 锦上添花
- 八答晕锦：{龟背纹(骨架), 牡丹, 宝相花, 联珠纹} 几何骨架填花
- 落花流水锦：{梅花, 流水纹, 卷草} 梅花漂浮水波，文人雅致
- 灯笼锦：{灯笼(主纹), 流苏, 花卉} 又名庆丰年、天下乐
- 陵阳公样：{联珠团窠, 对雉/对马/翔凤} 波斯萨珊与中国传统融合
- 十样锦：长安竹、天下乐、雕团、宜男等十种经典锦样合称
- 百子图锦：{儿童嬉戏} 多子多福

请按以下 JSON 格式输出（只输出 JSON，不要其他文字）：
{
  "craft_type": "蜀锦或蜀绣",
  "motifs": [
    {
      "name": "纹样名称（必须从上表中选择）",
      "category": "纹样类别",
      "role": "主纹样/底纹/边框纹/辅助纹/点缀纹",
      "position": "中心/满铺/边框/散点/角隅",
      "layout": "中心对称/二方连续/四方连续/角隅/散点/团窠",
      "coverage_ratio": 0.0到1.0之间的比例,
      "confidence": 0.0到1.0之间的置信度,
      "cultural_meaning": ["该纹样在图中体现的文化含义1", "含义2"],
      "description": "对该纹样在图中具体形态的简要描述"
    }
  ],
  "composition": {
    "primary_layout": "主要构图类型",
    "rhythm": "构图节奏描述",
    "description": "整体构图分析"
  },
  "composite_pattern_hint": "根据纹样组合推测最可能的复合纹样类型（如月华锦、凤穿牡丹等；若无法判断则填'不确定'）",
  "cultural_impression": "整体文化感受的一段话"
}"""

MOTIF_DETAIL_PROMPT = """你是一位蜀锦/蜀绣文化研究专家。对于图中展示的纹样"{}"，请分析其文化内涵。

请按 JSON 格式输出：
{
  "cultural_meanings": ["含义1", "含义2", ...],
  "origin": "历史来源",
  "iconic_form": "典型形态描述",
  "typical_colors": ["颜色1", ...],
  "narrative": "一段完整的文化叙事，解释该纹样在蜀锦中的意义"
}"""

COMPOSITION_PROMPT = """你是一位纺织品纹样构图分析专家。请分析这幅织锦/刺绣图像的构图特征。

参考构图类型：中心对称、二方连续、四方连续、角隅纹样、散点分布、团窠纹、陵阳公样、满地铺陈、折枝花、流水纹

请按 JSON 格式输出：
{
  "layouts": [{"type": "构图类型", "region": "应用区域", "description": "描述"}],
  "rhythm": "总体节奏感描述",
  "transformation_types": ["简化/夸张/旋转/缩放/抽象化/组合/叠晕/几何骨架"],
  "analysis": "完整的构图分析"
}"""


# ═══════════════════════════════════════════════════════════════
# VL Recognizer Class
# ═══════════════════════════════════════════════════════════════

class VLMotifRecognizer:
    """Zero-shot motif recognition using Qwen2.5-VL via DashScope API."""

    def __init__(self, base_url: str = None, api_key: str = None,
                 model: str = None):
        self.config = {
            "base_url": base_url or QWEN_VL_CONFIG["base_url"],
            "api_key": api_key or QWEN_VL_CONFIG["api_key"],
            "model": model or QWEN_VL_CONFIG["model"],
        }
        self.client = OpenAI(
            base_url=self.config["base_url"],
            api_key=self.config["api_key"],
        )
        self.model = self.config["model"]

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 data URL."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_vl(self, image_path: str, prompt: str,
                 temperature: float = 0.3) -> str:
        """Call Qwen2.5-VL with an image and text prompt."""
        ext = Path(image_path).suffix.lower()
        mime = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(ext, "image/jpeg")

        b64 = self._encode_image(image_path)
        data_url = f"data:{mime};base64,{b64}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    def _parse_json(self, text: str) -> dict:
        """Robust JSON parsing from VL response."""
        text = text.strip()
        # Remove markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON block
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"raw_response": text, "parse_error": True}

    # ── Recognition Tasks ──

    def decompose_motifs(self, image_path: str) -> dict:
        """Task 1: Identify all motifs in the image with roles and positions."""
        print(f"  [VL] 纹样分解: {Path(image_path).name}")
        t0 = time.time()
        response = self._call_vl(image_path, MOTIF_DECOMPOSITION_PROMPT)
        result = self._parse_json(response)
        result["_vl_response"] = response
        result["_elapsed"] = round(time.time() - t0, 1)
        return result

    def motif_detail(self, image_path: str, motif_name: str) -> dict:
        """Task 2: Analyze cultural meaning of a specific motif."""
        print(f"  [VL] 纹样文化分析: {motif_name}")
        prompt = MOTIF_DETAIL_PROMPT.format(motif_name)
        response = self._call_vl(image_path, prompt)
        return self._parse_json(response)

    def analyze_composition(self, image_path: str) -> dict:
        """Task 3: Analyze composition grammar of the image."""
        print(f"  [VL] 构图分析: {Path(image_path).name}")
        t0 = time.time()
        response = self._call_vl(image_path, COMPOSITION_PROMPT)
        result = self._parse_json(response)
        result["_elapsed"] = round(time.time() - t0, 1)
        return result

    # ── Full Pipeline ──

    def recognize(self, image_path: str,
                  do_detail: bool = False,
                  do_composition: bool = False) -> dict:
        """Run full recognition pipeline on an image.

        Returns structured dict compatible with annotation format.
        """
        result = {
            "image_path": str(image_path),
            "filename": Path(image_path).name,
        }

        # Task 1: Motif decomposition (always)
        decomp = self.decompose_motifs(image_path)
        result["motif_decomposition"] = decomp

        # Task 2: Detail analysis per motif (optional)
        if do_detail and "motifs" in decomp:
            details = []
            for m in decomp.get("motifs", []):
                if m.get("name"):
                    detail = self.motif_detail(image_path, m["name"])
                    details.append({"motif": m["name"], "detail": detail})
            result["motif_details"] = details

        # Task 3: Composition analysis (optional)
        if do_composition:
            result["composition_analysis"] = self.analyze_composition(image_path)

        return result

    def enrich_with_kg(self, vl_result: dict, kg) -> dict:
        """Enrich VL recognition results with KG knowledge.

        For each recognized motif:
        - Look up in ATOMIC_MOTIFS for cultural meanings, origin, etc.
        - Find composite patterns containing this motif
        - Cross-validate VL meanings vs KG meanings
        """
        enriched = {
            "image": vl_result.get("image_path", ""),
            "vl_recognition": vl_result.get("motif_decomposition", {}),
        }

        motifs = vl_result.get("motif_decomposition", {}).get("motifs", [])
        enriched_motifs = []

        for m in motifs:
            name = m.get("name", "")
            motif_key = self._match_motif_key(name)

            kg_entry = kg.motifs.get(motif_key, {})
            vl_meanings = set(m.get("cultural_meaning", []))
            kg_meanings = set(kg_entry.get("cultural_meanings", []))

            enriched_motifs.append({
                "vl_name": name,
                "vl_role": m.get("role", ""),
                "vl_position": m.get("position", ""),
                "vl_layout": m.get("layout", ""),
                "vl_coverage": m.get("coverage_ratio", 0),
                "vl_confidence": m.get("confidence", 0),
                "vl_description": m.get("description", ""),
                # KG enrichment
                "kg_name": kg_entry.get("name_zh", name),
                "kg_category": kg_entry.get("category", ""),
                "kg_meanings": list(kg_meanings),
                "kg_origin": kg_entry.get("origin", ""),
                "kg_description": kg_entry.get("description", "")[:200],
                # Cross-validation
                "meanings_overlap": list(vl_meanings & kg_meanings),
                "meanings_vl_only": list(vl_meanings - kg_meanings),
                "meanings_kg_only": list(kg_meanings - vl_meanings),
            })

        # Find matching composite patterns
        recognized_keys = {self._match_motif_key(m.get("name", ""))
                          for m in motifs if m.get("name")}
        pattern_matches = []
        for pid, pattern in kg.patterns.items():
            pattern_keys = {pm["motif"] for pm in pattern["motifs"]}
            overlap = recognized_keys & pattern_keys
            if overlap:
                match_ratio = len(overlap) / len(pattern_keys)
                pattern_matches.append({
                    "pattern_id": pid,
                    "name": pattern["name_zh"],
                    "craft": pattern.get("craft", ""),
                    "era": pattern.get("era", ""),
                    "match_ratio": round(match_ratio, 2),
                    "matched": [kg.motifs.get(k, {}).get("name_zh", k) for k in overlap],
                    "missing": [kg.motifs.get(k, {}).get("name_zh", k) for k in pattern_keys - overlap],
                    "narrative": pattern.get("cultural_narrative", ""),
                })
        pattern_matches.sort(key=lambda x: x["match_ratio"], reverse=True)

        enriched["enriched_motifs"] = enriched_motifs
        enriched["suggested_patterns"] = pattern_matches

        # KG stats from VL result
        n_with_kg = sum(1 for em in enriched_motifs if em["kg_category"])
        enriched["kg_coverage"] = f"{n_with_kg}/{len(enriched_motifs)} motifs mapped to KG"

        return enriched

    def _match_motif_key(self, name: str) -> str:
        """Match Chinese motif name to KG key."""
        mapping = {
            "凤凰纹": "phoenix", "龙纹": "dragon", "蝴蝶纹": "butterfly",
            "麒麟纹": "qilin", "仙鹤纹": "crane", "鸳鸯纹": "mandarin_duck",
            "孔雀纹": "peacock", "鹿纹": "deer",
            "狮纹": "lion",
            "卷草纹": "scroll_grass", "牡丹纹": "peony", "莲花纹": "lotus",
            "宝相花纹": "baoxiang_flower", "梅花纹": "plum_blossom",
            "云纹": "cloud", "万字纹": "swastika", "回纹": "meander",
            "龟背纹": "turtle_shell", "联珠纹": "pearl_roundel",
            "月华纹": "moon_halo",
            "文字纹": "auspicious_char",
        }
        # Also try simplified name matching
        for key, val in mapping.items():
            if val == name or key == name:
                return val
        return mapping.get(name, name.lower().replace(" ", "_"))

    # ── End-to-End Pipeline ──

    def recognize_and_understand(self, image_path: str, kg=None,
                                  do_composition: bool = True) -> dict:
        """Full end-to-end pipeline: VL recognition → KG enrichment → narrative.

        Pipeline stages:
          1. Qwen2.5-VL zero-shot motif decomposition
          2. KG enrichment (cultural meanings, origin, pattern matching)
          3. Cross-validation (VL vs KG knowledge consistency)
          4. Composite pattern inference
          5. Cultural narrative synthesis

        Args:
            image_path: Path to the textile image
            kg: ThreeLayerKG or EnhancedThreeLayerKG instance (auto-creates if None)
            do_composition: Also run composition analysis

        Returns:
            {
                "image": str,
                "craft_type": str,
                "motifs": [enriched motif dicts],
                "composition": dict,
                "suggested_patterns": [matched patterns],
                "cultural_narrative": str,
                "cross_validation": dict,
                "report": str (formatted text),
            }
        """
        from .three_layer_kg import ThreeLayerKG
        if kg is None:
            kg = ThreeLayerKG()

        print(f"\n{'='*64}")
        print(f"端到端识别流水线")
        print(f"{'='*64}")
        print(f"  图像: {Path(image_path).name}")

        # ═══ Stage 1: VL Decomposition ═══
        print(f"\n  [Stage 1/4] Qwen2.5-VL 纹样分解...")
        vl_result = self.recognize(image_path, do_composition=do_composition)
        decomp = vl_result.get("motif_decomposition", {})
        motifs = decomp.get("motifs", [])
        composition = decomp.get("composition", {})
        cultural_impression = decomp.get("cultural_impression", "")

        print(f"    → 识别到 {len(motifs)} 个纹样: "
              f"{', '.join(m.get('name','?') for m in motifs)}")
        print(f"    → 构图: {composition.get('primary_layout', '?')}")
        print(f"    → VL耗时: {decomp.get('_elapsed', '?')}s")

        # ═══ Stage 2: KG Enrichment ═══
        print(f"\n  [Stage 2/4] KG 知识增强...")
        enriched = self.enrich_with_kg(vl_result, kg)

        n_kg = sum(1 for em in enriched["enriched_motifs"] if em["kg_category"])
        print(f"    → {n_kg}/{len(motifs)} 纹样匹配到KG")

        # ═══ Stage 3: Cross-Validation ═══
        print(f"\n  [Stage 3/4] 交叉验证 (VL ↔ KG)...")
        vl_only_insights = []
        kg_only_insights = []
        confirmed = []

        for em in enriched["enriched_motifs"]:
            if em["meanings_overlap"]:
                confirmed.append(f"{em['vl_name']}: {', '.join(em['meanings_overlap'])}")
            if em["meanings_vl_only"]:
                vl_only_insights.append(f"{em['vl_name']}: {', '.join(em['meanings_vl_only'])}")
            if em["meanings_kg_only"]:
                kg_only_insights.append(f"{em['vl_name']}: {', '.join(em['meanings_kg_only'])}")

        cross_val = {
            "confirmed": confirmed,
            "vl_only": vl_only_insights,
            "kg_only": kg_only_insights,
        }

        if confirmed:
            print(f"    ✓ 一致: {len(confirmed)} 项")
            for c in confirmed:
                print(f"      {c}")
        if vl_only_insights:
            print(f"    ⚡ VL新发现 (建议补充KG): {len(vl_only_insights)} 项")
            for v in vl_only_insights:
                print(f"      → {v}")
        if kg_only_insights:
            print(f"    📚 KG补充 (VL未覆盖): {len(kg_only_insights)} 项")
            for k in kg_only_insights:
                print(f"      → {k}")

        # ═══ Stage 4: Pattern Matching & Narrative ═══
        print(f"\n  [Stage 4/4] 复合纹样匹配 & 文化叙事合成...")
        patterns = enriched.get("suggested_patterns", [])

        if patterns:
            top = patterns[0]
            print(f"    → 最佳匹配: {top['name']} ({top['craft']}, {top['era']}) "
                  f"匹配率={top['match_ratio']}")
            if len(patterns) > 1:
                print(f"    → 候选: {', '.join(p['name'] for p in patterns[1:3])}")

        # Synthesize narrative
        narrative = self._synthesize_narrative(
            enriched["enriched_motifs"], patterns, composition, cultural_impression
        )

        # ═══ Build report ═══
        report = self._format_report(
            image_path, enriched["enriched_motifs"], composition,
            patterns, cultural_impression, cross_val,
            decomp.get("_elapsed", 0)
        )

        print(f"\n  [完成] 端到端流水线结束")
        print(f"{'='*64}")

        return {
            "image": str(image_path),
            "craft_type": decomp.get("craft_type", ""),
            "motifs": enriched["enriched_motifs"],
            "composition": composition,
            "cultural_impression": cultural_impression,
            "suggested_patterns": patterns,
            "cross_validation": cross_val,
            "narrative": narrative,
            "report": report,
        }

    def _synthesize_narrative(self, enriched_motifs: list, patterns: list,
                               composition: dict, vl_impression: str) -> str:
        """Synthesize a comprehensive cultural narrative."""
        lines = []

        # Motif summary
        motif_names = [em.get("kg_name", em.get("vl_name", "?"))
                       for em in enriched_motifs]
        lines.append("、".join(motif_names[:4]))
        if len(motif_names) > 4:
            lines[-1] += f"等{len(motif_names)}种纹样"

        # Add cultural meanings
        meanings_list = []
        for em in enriched_motifs:
            if em.get("kg_meanings"):
                for m in em["kg_meanings"]:
                    if m not in meanings_list:
                        meanings_list.append(m)
        if meanings_list:
            lines.append(f"寓意{', '.join(meanings_list[:5])}")

        # Add composition
        if composition:
            layout = composition.get("primary_layout", "")
            if layout:
                lines.append(f"采用{layout}构图")

        # Add VL impression
        if vl_impression:
            lines.append(vl_impression)

        # Add pattern match
        if patterns and patterns[0]["match_ratio"] >= 0.3:
            p = patterns[0]
            lines.append(f"符合{p['name']}({p.get('craft','')})的典型纹样配置")

        return "。".join(line for line in lines if line) + "。"

    def _format_report(self, image_path: str, enriched_motifs: list,
                       composition: dict, patterns: list,
                       vl_impression: str, cross_val: dict,
                       vl_elapsed: float) -> str:
        """Format a structured analysis report."""
        lines = []
        lines.append("=" * 64)
        lines.append(f"蜀锦/蜀绣纹样智能识别分析报告")
        lines.append("=" * 64)
        lines.append(f"图像: {Path(image_path).name}")
        lines.append(f"识别引擎: Qwen2.5-VL (零样本)")
        lines.append(f"知识图谱: 三层KG (21纹样 × 10布局 × 12复合纹样)")
        lines.append(f"VL响应时间: {vl_elapsed}s")
        lines.append("")

        # ── Identified Motifs ──
        lines.append("─" * 40)
        lines.append(f"【纹样识别结果】({len(enriched_motifs)} 个纹样)")
        lines.append("─" * 40)
        for i, em in enumerate(enriched_motifs, 1):
            name = em.get("kg_name", em.get("vl_name", "?"))
            cat = em.get("kg_category", "")
            role = em.get("vl_role", "?")
            pos = em.get("vl_position", "?")
            layout = em.get("vl_layout", "?")
            cov = f"{em.get('vl_coverage', 0):.0%}"
            conf = f"{em.get('vl_confidence', 0):.0%}"

            lines.append(f"\n  {i}. {name} ({cat})")
            lines.append(f"     角色: {role} | 位置: {pos} | 布局: {layout}")
            lines.append(f"     占比: {cov} | 置信度: {conf}")
            if em.get("kg_origin"):
                lines.append(f"     历史来源: {em['kg_origin']}")
            if em.get("kg_meanings"):
                lines.append(f"     文化含义: {', '.join(em['kg_meanings'])}")
            if em.get("vl_description"):
                lines.append(f"     VL描述: {em['vl_description']}")

        # ── Composition ──
        if composition:
            lines.append(f"\n{'─'*40}")
            lines.append(f"【构图分析】")
            lines.append(f"{'─'*40}")
            lines.append(f"  主布局: {composition.get('primary_layout', '?')}")
            if composition.get("rhythm"):
                lines.append(f"  节奏: {composition['rhythm']}")
            if composition.get("description"):
                lines.append(f"  描述: {composition['description']}")

        # ── Cross-Validation ──
        lines.append(f"\n{'─'*40}")
        lines.append(f"【VL ↔ KG 交叉验证】")
        lines.append(f"{'─'*40}")
        if cross_val.get("confirmed"):
            lines.append(f"  ✓ 知识一致 ({len(cross_val['confirmed'])}项):")
            for c in cross_val["confirmed"]:
                lines.append(f"    {c}")
        if cross_val.get("vl_only"):
            lines.append(f"  ⚡ VL新发现 → 建议补充KG ({len(cross_val['vl_only'])}项):")
            for v in cross_val["vl_only"]:
                lines.append(f"    {v}")
        if cross_val.get("kg_only"):
            lines.append(f"  📚 KG补充 → VL未覆盖 ({len(cross_val['kg_only'])}项):")
            for k in cross_val["kg_only"]:
                lines.append(f"    {k}")

        # ── Pattern Matching ──
        if patterns:
            lines.append(f"\n{'─'*40}")
            lines.append(f"【复合纹样匹配】")
            lines.append(f"{'─'*40}")
            for p in patterns[:3]:
                status = "★ 最佳匹配" if p == patterns[0] else "  候选"
                lines.append(f"  {status}: {p['name']} ({p.get('craft','')}, {p.get('era','')})")
                lines.append(f"    匹配率: {p['match_ratio']:.0%}")
                lines.append(f"    已匹配: {', '.join(p['matched'])}")
                if p.get("missing"):
                    lines.append(f"    缺失纹样: {', '.join(p['missing'])}")

        # ── Cultural Impression ──
        if vl_impression:
            lines.append(f"\n{'─'*40}")
            lines.append(f"【文化感受】")
            lines.append(f"{'─'*40}")
            lines.append(f"  {vl_impression}")

        lines.append(f"\n{'='*64}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════

def _find_test_image() -> Optional[str]:
    """Find a test image from seeded motif directories."""
    motifs_dir = Path(__file__).parent.parent / "data" / "images" / "motifs"
    if not motifs_dir.exists():
        return None
    for motif_dir in sorted(motifs_dir.iterdir()):
        if not motif_dir.is_dir():
            continue
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            for img in sorted(motif_dir.glob(ext)):
                return str(img)
    return None


def main():
    from .three_layer_kg import ThreeLayerKG

    # Find test image
    test_img = _find_test_image()
    if not test_img:
        print("没有找到测试图像。请先在 data/images/motifs/ 放置纹样图像。")
        return

    # Init recognizer and KG
    recognizer = VLMotifRecognizer()
    kg = ThreeLayerKG()

    # Run full end-to-end pipeline
    result = recognizer.recognize_and_understand(test_img, kg=kg)

    # Print formatted report
    print(result["report"])


if __name__ == "__main__":
    main()
