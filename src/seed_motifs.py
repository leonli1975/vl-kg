"""Batch seed motif visual centroids from per-motif image directories.

Workflow:
  1. Place reference images in data/images/motifs/<纹样名>/
  2. Run: python -m src.seed_motifs [--incremental]
  3. Script extracts ResNet18 + CN-CLIP features and seeds centroids
  4. Reports: seeded / missing / total per motif

Directory mapping:
  data/images/motifs/
    ├── 凤凰纹/    → motif_id: phoenix
    ├── 龙纹/      → motif_id: dragon
    ├── ...
    └── 文字纹/    → motif_id: auspicious_char
"""

import json, os, sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from .enhanced_kg import EnhancedThreeLayerKG
from .three_layer_kg import ATOMIC_MOTIFS


MOTIFS_DIR = Path(__file__).parent.parent / "data" / "images" / "motifs"
STATE_FILE = Path(__file__).parent.parent / "data" / "motif_seed_state.json"

# Mapping: Chinese directory name → motif key in the KG
DIR_TO_MOTIF = {
    "凤凰纹": "phoenix",
    "龙纹": "dragon",
    "蝴蝶纹": "butterfly",
    "麒麟纹": "qilin",
    "仙鹤纹": "crane",
    "鸳鸯纹": "mandarin_duck",
    "孔雀纹": "peacock",
    "鹿纹": "deer",
    "狮纹": "lion",
    "卷草纹": "scroll_grass",
    "牡丹纹": "peony",
    "莲花纹": "lotus",
    "宝相花纹": "baoxiang_flower",
    "梅花纹": "plum_blossom",
    "云纹": "cloud",
    "万字纹": "swastika",
    "回纹": "meander",
    "龟背纹": "turtle_shell",
    "联珠纹": "pearl_roundel",
    "文字纹": "auspicious_char",
}

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def scan_motif_dirs() -> Dict[str, List[Path]]:
    """Scan per-motif directories and return {motif_id: [image_paths]}."""
    found = {}
    if not MOTIFS_DIR.exists():
        print(f"[ERROR] Motif images directory not found: {MOTIFS_DIR}")
        print(f"  Create it with: mkdir -p {MOTIFS_DIR}/{{凤凰纹,龙纹,...}}")
        return found

    for dir_name, motif_id in DIR_TO_MOTIF.items():
        dir_path = MOTIFS_DIR / dir_name
        if not dir_path.is_dir():
            continue
        images = []
        for ext in VALID_EXTS:
            images.extend(sorted(dir_path.glob(f"*{ext}")))
        if images:
            found[motif_id] = images
    return found


def load_state() -> dict:
    """Load previous seeding state (which images were used)."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    """Save seeding state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def seed_all(incremental: bool = False):
    """Main seeding pipeline."""
    print("=" * 64)
    print("纹样视觉质心批量播种")
    print("=" * 64)

    # ── Scan ──
    images_by_motif = scan_motif_dirs()
    if not images_by_motif:
        print("\n[!] No motif images found.")
        print(f"  Place images in: {MOTIFS_DIR}/")
        print(f"  One subdirectory per motif, e.g.:")
        print(f"    {MOTIFS_DIR}/凤凰纹/phoenix_01.jpg")
        print(f"    {MOTIFS_DIR}/龙纹/dragon_01.jpg")
        return

    print(f"\n[1] 扫描结果: {len(images_by_motif)} 个纹样目录有图像")
    for motif_id, paths in sorted(images_by_motif.items()):
        name = ATOMIC_MOTIFS.get(motif_id, {}).get("name_zh", motif_id)
        print(f"    {name}: {len(paths)} 张")

    # ── Init KG ──
    print(f"\n[2] 初始化增强KG...")
    kg = EnhancedThreeLayerKG()

    # ── Load state ──
    state = load_state()
    if incremental:
        print(f"   增量模式: 已加载 {len(state)} 个纹样的播种状态")

    # ── Seed ──
    print(f"\n[3] 播种视觉质心...")
    seeded = []
    skipped = []
    for motif_id, img_paths in sorted(images_by_motif.items()):
        name = ATOMIC_MOTIFS.get(motif_id, {}).get("name_zh", motif_id)
        str_paths = [str(p) for p in img_paths]

        if incremental and motif_id in state:
            prev_paths = set(state[motif_id].get("paths", []))
            new_paths = [p for p in str_paths if p not in prev_paths]
            if not new_paths:
                skipped.append((name, len(str_paths)))
                continue
            str_paths = new_paths

        try:
            kg.seed_motif(motif_id, str_paths)
            seeded.append((name, len(str_paths)))
            state[motif_id] = {
                "name_zh": name,
                "paths": [str(p) for p in img_paths],
                "n_images": len(img_paths),
            }
            print(f"  ✓ {name}: {len(str_paths)} 张 → 质心已更新")
        except Exception as e:
            print(f"  ✗ {name}: 播种失败 - {e}")

    if skipped:
        print(f"\n  跳过 (无新增图像):")
        for name, n in skipped:
            print(f"    {name}: {n} 张 (已播种)")

    # ── Save state ──
    save_state(state)

    # ── Report ──
    st = kg.stats()
    print(f"\n[4] 播种结果")
    print(f"  已播种纹样: {st['motifs_with_visual_centroids']}/{st['motifs']}")
    print(f"  视觉样本总数: {st['total_visual_samples']}")

    # ── Missing motifs ──
    all_motif_ids = set(ATOMIC_MOTIFS.keys())
    seeded_ids = set(images_by_motif.keys())
    missing = all_motif_ids - seeded_ids
    if missing:
        print(f"\n[5] 待播种纹样 (未找到图像)")
        for mid in sorted(missing):
            name = ATOMIC_MOTIFS.get(mid, {}).get("name_zh", mid)
            cat = ATOMIC_MOTIFS.get(mid, {}).get("category", "?")
            dir_name = {v: k for k, v in DIR_TO_MOTIF.items()}.get(mid, mid)
            print(f"  ⧖ {name} ({cat}) → 请在 {MOTIFS_DIR / dir_name}/ 放置图像")

    # ── Quick retrieval test ──
    if seeded:
        print(f"\n[6] 快速检索测试")
        for motif_id in sorted(seeded_ids)[:3]:
            name = ATOMIC_MOTIFS.get(motif_id, {}).get("name_zh", motif_id)
            desc = ATOMIC_MOTIFS.get(motif_id, {}).get("description", "")
            # Text→motif test
            results = kg.query_text(desc[:60], top_k=3)
            rank = next((i+1 for i, (mid, _) in enumerate(results) if mid == motif_id), None)
            if rank:
                print(f"  ✓ {name}: 文本检索排名 #{rank}")

    print(f"\n{'='*64}")
    print("播种完成。运行 python -m src.enhanced_kg 验证。")
    print(f"{'='*64}")


def show_status():
    """Show current seeding status without modifying anything."""
    kg = EnhancedThreeLayerKG()
    st = kg.stats()
    all_ids = set(ATOMIC_MOTIFS.keys())

    print("=" * 64)
    print("纹样播种状态")
    print("=" * 64)
    print(f"  已播种: {st['motifs_with_visual_centroids']}/{st['motifs']}")
    print(f"  视觉样本: {st['total_visual_samples']}")
    print()

    for mid in sorted(all_ids):
        store = kg.motif_embeddings.get(mid, {})
        n = store.get("n", 0)
        name = ATOMIC_MOTIFS.get(mid, {}).get("name_zh", mid)
        cat = ATOMIC_MOTIFS.get(mid, {}).get("category", "?")
        status = f"✓ {n}张" if n > 0 else "⧖ 无图像"
        print(f"  [{status}] {name} ({cat})")
    print(f"\n{'='*64}")


def main():
    if "--status" in sys.argv:
        show_status()
    else:
        incremental = "--incremental" in sys.argv
        seed_all(incremental=incremental)


if __name__ == "__main__":
    main()
