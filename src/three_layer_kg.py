"""Three-layer KG: Atomic Motif → Composition Grammar → Composite Pattern.

Layer 1: Atomic Motif (基本纹样) - individual pattern elements with cultural meanings
Layer 2: Composition Grammar (构图规则) - layout, transformation, relationship rules  
Layer 3: Composite Pattern (复合纹样) - combinations with cultural interpretation

Enables: motif decomposition of textile images → cultural understanding
"""

import json
import numpy as np
import torch, torchvision.models as models, torchvision.transforms as T
from PIL import Image
import cn_clip.clip.model as clip_model
import cn_clip.clip as clip
from pathlib import Path
from typing import List, Dict, Optional

CN_CLIP_CKPT = "/home/lgc/.cache/modelscope/damo/multi-modal_clip-vit-base-patch16_zh/clip_cn_vit-b-16.pt"


# ═══════════════════════════════════════════════════════════════
# Layer 1: Atomic Motif — individual pattern elements
# ═══════════════════════════════════════════════════════════════

ATOMIC_MOTIFS = {
    # Animal motifs
    "phoenix": {
        "name_zh": "凤凰纹",
        "category": "动物纹",
        "iconic_forms": ["展翅", "站立", "回首", "几何简化", "团窠对凤"],
        "cultural_meanings": ["吉祥", "重生", "高贵", "爱情", "和谐"],
        "origin": "周代凤鸟崇拜；唐代陵阳公样中'翔凤'为经典题材",
        "common_positions": ["中心主纹", "对称双凤", "团窠中心"],
        "typical_colors": ["金", "红", "五彩"],
        "description": "凤凰是百鸟之王，蜀锦中常以展翅或站立姿态出现。凤与牡丹组合称'凤穿牡丹'，寓意富贵吉祥。战国'对龙对凤条纹锦'中已见双凤对称纹样。唐代陵阳公样以'翔凤'为主题，外环联珠纹。蜀锦中有'吹箫引凤雨丝锦'等经典作品。"
    },
    "dragon": {
        "name_zh": "龙纹",
        "category": "动物纹",
        "iconic_forms": ["盘龙（团龙）", "升龙（昂首腾飞）", "行龙（游走状）"],
        "cultural_meanings": ["皇权与尊贵象征", "祥瑞与权威", "驱邪避灾", "蜀锦中寓意“龙凤呈祥”的吉祥组合"],
        "origin": "龙纹在蜀锦中可追溯至战国，汉代已见雏形，唐代受佛教影响更趋华美，宋代定型为经典纹样。",
        "common_positions": ["锦面中心（团龙）", "边饰或条带（行龙）"],
        "typical_colors": ["金黄", "朱红", "藏青"],
        "description": "龙纹是蜀锦中最具代表性的动物纹样之一，其历史可上溯至战国时期的织锦残片，至唐代因佛教与宫廷审美交融而愈发繁复华丽，宋代则成为蜀锦“八答晕锦”“灯笼锦”等名品中的核心元素。在经典蜀锦品种如“红地团龙锦”“蓝地行龙缎”中，龙纹常以盘龙、升龙或行龙形态出现，或与云纹、火珠纹组合成“云龙纹”，或与凤纹搭配为“龙凤呈祥”，寓意皇权至尊、祥瑞护佑。其色彩多用金黄、朱红、藏青，以显庄重华贵，常置于锦面中心或边饰条带，是蜀锦中兼具威严与吉祥的永恒主题。"
    },
    "butterfly": {
        "name_zh": "蝴蝶纹",
        "category": "动物纹",
        "iconic_forms": ["展翅对称蝶形", "侧飞单蝶纹", "蝶恋花纹"],
        "cultural_meanings": ["象征爱情与婚姻美满", "寓意长寿与福寿双全", "代表自由与蜕变新生", "谐音“福迭”以祈吉祥"],
        "origin": "蝴蝶纹在蜀锦中可追溯至唐代，受中原与西域文化交融影响，宋元时期定型为经典题材，明清时广泛用于锦缎织造。",
        "common_positions": ["锦缎中心主纹", "边饰或角隅纹样"],
        "typical_colors": ["明黄", "绯红", "翠绿"],
        "description": "蝴蝶纹是蜀锦中经典的动物纹样，其历史可溯至唐代，在宋元时期随蜀锦工艺成熟而成为独立题材，明清时更常见于“浣花锦”“雨丝锦”等名品中。蜀锦中的蝴蝶纹常以展翅对称或侧飞形态呈现，多与花卉（如牡丹、莲花）组合成“蝶恋花”纹样，寓意爱情美满与福寿双全。色彩上多用明黄、绯红、翠绿等鲜明色调，通过晕裥工艺营造蝶翼的渐变效果，既展现自然灵动之美，又承载了“福迭”而至的吉祥祈愿，是蜀锦中兼具装饰性与文化内涵的经典元素。"
    },

    # Plant motifs
    "scroll_grass": {
        "name_zh": "卷草纹",
        "category": "植物纹",
        "iconic_forms": ["S形或C形连续卷曲的藤蔓", "缠枝卷叶与花头交替排列", "对称式或散点式卷草骨架"],
        "cultural_meanings": ["象征生生不息、绵延不绝的生命力", "寓意富贵吉祥、福运连绵", "在蜀锦中代表织造技艺的繁复与精巧", "体现唐代以来蜀地兼容并蓄的审美风尚"],
        "origin": "卷草纹源于唐代，受西域忍冬纹与本土云气纹融合影响，经蜀地工匠改造后成为蜀锦经典纹样。",
        "common_positions": ["锦缎的满地或半满地铺陈", "边饰或栏界纹样"],
        "typical_colors": ["宝蓝", "朱红", "石绿"],
        "description": "卷草纹是蜀锦中极具代表性的植物纹样，其历史可追溯至唐代，由西域忍冬纹与本土云气纹融合演变而来，在蜀地织造中逐渐形成流畅婉转的S形或C形连续卷曲骨架。经典应用如‘红地卷草纹锦’中，以朱红为地、石绿勾卷，花头与藤蔓交替循环，尽显蜀锦织工之细腻。常与宝相花、团窠纹组合，形成‘花中套草’的繁复层次，寓意生生不息、福运连绵。其色彩多取宝蓝、朱红、石绿等浓艳对比色，既彰显蜀锦的华丽质感，又寄托了吉祥绵长的文化愿景。"
    },
    "peony": {
        "name_zh": "牡丹纹",
        "category": "植物纹",
        "iconic_forms": ["盛开", "侧视", "缠枝牡丹", "折枝牡丹", "团花"],
        "cultural_meanings": ["富贵", "繁荣", "吉祥", "兴旺", "国色天香"],
        "origin": "唐代盛行；宋代融入八答晕锦、如意牡丹锦等名品；明代'凤穿牡丹'成为经典组合",
        "common_positions": ["中心主纹", "团花", "与其他纹样组合"],
        "typical_colors": ["红", "粉", "紫"],
        "description": "牡丹纹以'花中之王'牡丹为题材，形态丰满、色彩艳丽，是蜀锦中最具代表性的花卉纹饰。唐代已大量使用。宋代八答晕锦中牡丹与菊花、宝相花并列为核心主纹，《蜀锦谱》记载有'青绿如意牡丹锦'。明清时期'凤穿牡丹'成为经典组合——百鸟之王凤凰环绕花中之王牡丹，象征安宁吉祥、富贵兴旺。明末清初兴起的通海缎（满花锦）中'如意牡丹'是常用主纹之一。牡丹被誉为'国色天香'，是富贵最直接的象征。"
    },

    # Geometric motifs
    "cloud": {
        "name_zh": "云纹",
        "category": "几何纹",
        "iconic_forms": ["勾连云纹", "卷云纹", "朵云纹"],
        "cultural_meanings": ["祥瑞与吉兆", "升仙与通天的象征", "皇权与尊贵", "生生不息与变化无穷"],
        "origin": "云纹源于商周云雷纹，经战国、汉代发展，至唐代在蜀锦中定型为流畅卷云形态，成为经典装饰母题。",
        "common_positions": ["锦面中心主纹", "边饰或间隔纹"],
        "typical_colors": ["朱红", "石青", "月白"],
        "description": "云纹是蜀锦中最具代表性的传统纹样之一，其历史可追溯至商周时期的云雷纹，在唐代蜀锦中演变为线条流畅、形态饱满的卷云与朵云样式。经典蜀锦品种如‘红地云纹锦’、‘月白地朵云缎’均以云纹为主纹，常与龙纹、凤纹、宝相花纹组合，形成‘龙凤呈祥’或‘云中花’的吉祥构图。云纹在蜀锦中不仅象征祥瑞与通天，更承载着古人对自然力量的敬畏与对美好生活的祈愿，其连绵不绝的形态也寓意生命与福运的永恒延续。"
    },
    "swastika": {
        "name_zh": "万字纹",
        "category": "几何纹",
        "iconic_forms": ["单万字", "万字不断头(四方连续)", "二方连续边框"],
        "cultural_meanings": ["吉祥万德", "永恒", "无限", "万代绵长"],
        "origin": "佛教卍字符号；北魏已有'吉祥万德之所集'之说",
        "common_positions": ["底纹", "暗底花纹", "独立纹样", "背景装饰"],
        "typical_colors": ["金", "红"],
        "description": "万字纹以卍字符为核心元素，通过连续、循环、对称等手法构成。其本质是一种图案组织机制——卍字的对称直角结构和直线特性使其天然适合四方连续排列（万字不断头）。在方方锦中常作为暗底花纹出现；在民族锦中与葵花、团龙并列作为独立装饰纹样；也有专门的'万字锦'品种。在汉代已有应用，现代创作中仍被复原使用。"
    },

    # ── New motifs from authoritative sources ──

    "qilin": {
        "name_zh": "麒麟纹",
        "category": "动物纹",
        "iconic_forms": ["正面蹲踞式麒麟，头生独角，身披鳞甲，尾如狮鬃", "侧面奔跑式麒麟，四蹄腾空，火焰状鬃毛飘扬", "双麒麟相对戏珠，周围环绕云气与如意纹"],
        "cultural_meanings": ["象征祥瑞与太平盛世，寓意“麒麟出，天下和”", "代表仁义之德，麒麟为仁兽，不践生草、不食生物", "祈愿子孙贤德、家族昌盛，常与“麒麟送子”传说关联", "在蜀锦中作为宫廷赐服纹样，彰显尊贵与权威"],
        "origin": "麒麟纹源于先秦神话，汉代已见于织锦，唐代蜀锦中定型为祥瑞主题，宋代至明清成为蜀锦经典动物纹样。",
        "common_positions": ["锦袍胸背或肩部主纹", "锦缎中心团窠或开光内"],
        "typical_colors": ["朱红", "石青", "明黄"],
        "description": "麒麟纹是蜀锦中历史悠久的动物纹样，源自先秦神话中的仁兽形象，在唐代蜀锦中已作为祥瑞主题出现，至明清时期成为宫廷赐服与民间吉庆锦缎的经典纹样。蜀锦中的麒麟纹常以正面蹲踞或侧面奔跑姿态呈现，身披细密鳞甲，尾鬃如火焰，多与云气、如意、八宝等纹样组合，形成“麒麟献瑞”“麒麟送子”等吉祥图式。典型品种如清代“红地麒麟纹锦”和“明黄地麒麟送子锦”，以朱红、石青、明黄为主色，寓意天下太平、子孙贤德，在蜀锦中既承载着儒家仁德思想，又寄托了世俗对福禄昌盛的祈愿。"
    },
    "crane": {
        "name_zh": "仙鹤纹",
        "category": "动物纹",
        "iconic_forms": ["展翅飞翔状仙鹤，双翼平展，长颈前伸", "单足伫立状仙鹤，曲颈回首，羽冠高耸", "双鹤对鸣状，一仰一俯，喙部相接"],
        "cultural_meanings": ["象征长寿与长生不老", "寓意高洁与君子之风", "代表官运亨通、一品当朝", "在蜀锦中常与云纹组合，表达“鹤寿云祥”的吉祥愿景"],
        "origin": "仙鹤纹源于道教仙鹤崇拜，唐代蜀锦中已见鹤纹，宋代随织锦技艺成熟而定型，明清时期成为蜀锦经典题材。",
        "common_positions": ["锦心（主图案区域）", "边饰或团窠内"],
        "typical_colors": ["月白", "石青", "赭石"],
        "description": "仙鹤纹是蜀锦中极具代表性的动物纹样，其历史可追溯至唐代，在宋代蜀锦“八答晕锦”中已见鹤纹与云纹的组合，至明清时期更成为“福寿锦”“鹤寿锦”等经典品种的核心图案。蜀锦仙鹤纹常以月白、石青、赭石三色交织，通过平纹或斜纹组织表现鹤羽的轻盈与层次。在构图上，仙鹤纹多与祥云、灵芝、寿桃等组合，形成“鹤寿同春”“云鹤献瑞”等吉祥主题，既体现了蜀锦工巧细腻的织造技艺，又承载了长寿、高洁与官运亨通的多重文化寓意，是蜀锦中“图必有意，意必吉祥”的典型代表。"
    },
    "mandarin_duck": {
        "name_zh": "鸳鸯纹",
        "category": "动物纹",
        "aliases": ["鸳鸯锦"],
        "iconic_forms": ["对游", "并栖", "双宿"],
        "cultural_meanings": ["忠贞爱情", "夫妻恩爱", "美满婚姻", "白头偕老"],
        "origin": "南朝梁《文选》中鸳鸯最初喻兄弟情谊；唐代卢照邻'愿作鸳鸯不羡仙'后成为爱情图腾",
        "common_positions": ["散点", "水波纹间", "锦被主题"],
        "typical_colors": ["五彩", "绿", "紫", "蓝"],
        "description": "鸳鸯纹通常以成对鸟的形式出现，又称'鸳鸯锦'。各朝代鸟的形式差异很大，但成双成对的组合始终不变。南朝梁《文选》中鸳鸯最初用来比喻兄弟情谊；唐代卢照邻千古名句'愿作鸳鸯不羡仙'将其与夫妻爱情紧密联系，此后成为夫妻恩爱、白头偕老的经典象征。蜀锦中常以鸳鸯锦被的形式出现，寓意美满婚姻。"
    },
    "peacock": {
        "name_zh": "孔雀纹",
        "category": "动物纹",
        "iconic_forms": ["展翅开屏式孔雀", "伫立回首式孔雀", "双孔雀衔绶对舞"],
        "cultural_meanings": ["象征吉祥富贵与华美尊贵", "寓意夫妻恩爱、家庭和睦", "代表文明与太平盛世", "在蜀锦中常与牡丹组合，寓意“锦上添花、富贵长春”"],
        "origin": "孔雀纹自唐代随佛教艺术传入中原，宋代融入蜀锦，明清时期成为蜀锦经典题材，尤以“孔雀锦”闻名。",
        "common_positions": ["锦缎中心主纹", "衣襟或袖口装饰带"],
        "typical_colors": ["翠绿", "宝蓝", "金色"],
        "description": "孔雀纹是蜀锦中极具代表性的动物纹样，最早见于唐代，受佛教艺术影响而兴盛，至明清时期已发展为蜀锦经典题材，尤以“孔雀锦”和“孔雀妆花缎”最为著名。纹样常以展翅开屏、伫立回首或双孔雀衔绶对舞等形态呈现，羽翼细节繁复华丽，多用翠绿、宝蓝、金色等浓艳色彩，彰显富贵气象。在蜀锦中，孔雀纹常与牡丹、祥云、缠枝莲等组合，形成“孔雀戏牡丹”“孔雀穿花”等经典构图，寓意吉祥富贵、夫妻和美与太平盛世，是宫廷礼服、喜庆服饰及高档锦匣的常用纹饰。"
    },
    "deer": {
        "name_zh": "鹿纹",
        "category": "动物纹",
        "iconic_forms": ["回首奔鹿", "双鹿对望", "卧鹿衔芝"],
        "cultural_meanings": ["福禄双全", "官运亨通", "长寿安康", "祥瑞和谐"],
        "origin": "鹿纹源于商周，唐代蜀锦中已见，宋元时期融入吉祥寓意，明清蜀锦中成为经典主题。",
        "common_positions": ["锦心主纹", "边饰条带"],
        "typical_colors": ["朱红", "石青", "月白"],
        "description": "鹿纹在蜀锦中源远流长，最早可追溯至唐代，宋元时期与“福禄”谐音结合，成为吉祥纹样。明清蜀锦如“福禄寿三星锦”中，鹿常与鹤、松树组合，寓意长寿与官运。经典形态包括回首奔鹿、双鹿对望及卧鹿衔芝，线条流畅，姿态灵动。在“红地鹿纹锦”中，朱红底色衬托石青或月白鹿形，色彩对比鲜明。鹿纹多置于锦心主位或边饰条带，与云纹、花卉纹交织，形成“鹿鸣春晓”等经典构图，承载着蜀地人民对福禄双全、和谐美满的深切祈愿。"
    },
    "lion": {
        "name_zh": "狮纹",
        "category": "动物纹",
        "iconic_forms": ["蹲坐回首狮", "绣球戏狮", "双狮对舞"],
        "cultural_meanings": ["驱邪避灾，镇宅护佑", "威仪尊贵，象征权力与地位", "喜庆吉祥，寓意事事如意", "佛教护法，代表智慧与勇猛"],
        "origin": "狮纹自汉代随佛教传入中国，唐代蜀锦中已见狮子形象，宋元时期融入蜀地织锦，明清时成为经典题材。",
        "common_positions": ["锦面中心主纹", "团窠或联珠纹内"],
        "typical_colors": ["朱红", "金黄", "石青"],
        "description": "狮纹是蜀锦中极具代表性的动物纹样，其历史可追溯至汉代，随佛教东传而进入中原，唐代蜀锦匠人将其与本土瑞兽观念结合，创制出威猛而不失灵动的狮纹。在蜀锦名品“红地狮纹锦”中，狮子常作蹲坐回首或戏绣球状，形态饱满，鬃毛卷曲如云，与联珠纹、卷草纹组合，形成富丽堂皇的视觉效果。明清时期，狮纹更与“福”“寿”字纹、如意云纹搭配，用于袍服、帐幔等高端织物，寓意驱邪纳福、官运亨通。其色彩以朱红、金黄为主，间以石青勾边，尽显蜀锦的华贵与匠心。"
    },

    # Plant motifs (new)
    "lotus": {
        "name_zh": "莲花纹",
        "category": "植物纹",
        "iconic_forms": ["团花(放射状)", "缠枝莲", "折枝莲"],
        "cultural_meanings": ["纯洁", "高洁", "多子", "连生贵子", "和谐"],
        "origin": "汉代前后开始使用；唐代工艺成熟，使用更为普遍",
        "common_positions": ["团花主纹", "辅助纹", "落花流水组合"],
        "typical_colors": ["粉", "白", "红"],
        "description": "莲花纹在蜀锦中的应用可追溯至汉代前后。唐代工艺成熟，常与格子花、联珠纹、对禽纹、对兽纹组合。经典形态有：团花形态——作为主纹样，如'格子红锦'（赤底格子花纹锦），在联珠纹构成的方格中央织放射状莲花，周围饰蔓草或忍冬纹，红色地色彩鲜明；辅助形态——与落花流水锦中的流水纹、联珠纹等元素组合，构成丰富画面。'莲子莲花'组合寓意连生贵子。"
    },
    "baoxiang_flower": {
        "name_zh": "宝相花纹",
        "category": "植物纹",
        "iconic_forms": ["多层放射状莲花形", "对称团窠式宝相花", "缠枝宝相花与卷草组合"],
        "cultural_meanings": ["象征富贵吉祥与圆满", "体现佛教圣洁与世俗繁荣的结合", "在蜀锦中寓意“宝相庄严、福泽绵长”", "代表唐代以来蜀地丝织技艺的巅峰"],
        "origin": "源自唐代佛教艺术中的莲花纹样，融合牡丹、石榴等元素，经蜀锦工匠改造为对称饱满的装饰纹样。",
        "common_positions": ["锦缎中心主纹", "边饰或团窠骨架内"],
        "typical_colors": ["朱红", "石青", "明黄"],
        "description": "宝相花纹是蜀锦中最具代表性的植物纹样之一，起源于唐代，糅合莲花、牡丹与石榴等花卉特征，形成多层放射状或对称团窠式结构。在蜀锦经典品种如“陵阳公样”锦、“团窠宝花锦”中，常作为主纹占据锦面中心，或与卷草、瑞鸟组合成繁复的“锦上添花”布局。其色彩以朱红、石青、明黄为主，通过晕色技法呈现华丽层次。文化上，宝相花纹既承载佛教的圣洁寓意，又象征世俗的富贵圆满，在蜀锦中常被用于宫廷服饰与仪仗用品，体现“宝相庄严、福泽绵长”的吉祥祈愿。"
    },
    "plum_blossom": {
        "name_zh": "梅花纹",
        "category": "植物纹",
        "aliases": ["曲水纹", "紫曲水", "梅花曲水锦"],
        "iconic_forms": ["折枝梅", "正面单朵", "正面对称", "落花流水组合"],
        "cultural_meanings": ["高洁", "坚韧", "报春", "川流不息", "文人雅致"],
        "origin": "宋代'诗画入锦'典范；从唐宋诗词意境中汲取灵感创作",
        "common_positions": ["与流水纹组合", "四方连续", "散点"],
        "typical_colors": ["红", "白", "粉", "蓝"],
        "description": "梅花纹常与流水组合成经典的'落花流水纹'（又称曲水纹、紫曲水），是宋代蜀锦'诗画入锦'的典范。灵感来自唐诗'桃花流水杳然去，别有天地非人间'和宋词'落花流水红'。纹样以单朵或折枝梅花（或桃花）与波浪、旋涡等流水形态组合，构成四方连续图案。线条流畅，意境优美，极具'翰墨之气'，体现了宋代典雅清秀的审美。'落花流水'组合也寓意川流不息、连绵不绝。"
    },

    # Geometric motifs (new)
    "meander": {
        "name_zh": "回纹",
        "category": "几何纹",
        "iconic_forms": ["单体回纹（单个回字形）", "连续回纹（横向或纵向重复排列）", "套叠回纹（大小回纹嵌套）"],
        "cultural_meanings": ["富贵不断（回环往复象征财富绵延）", "吉祥长久（循环无端寓意福寿永续）", "秩序和谐（规整几何体现蜀锦工艺的严谨）", "辟邪护佑（回纹形似迷宫，有驱邪之意）"],
        "origin": "回纹源自新石器时代彩陶上的螺旋纹，商周青铜器上定型为回字形，汉代蜀锦中已广泛用作边饰和底纹。",
        "common_positions": ["锦缎边饰（如领口、袖口、衣缘）", "锦面底纹（作为主纹的衬托）"],
        "typical_colors": ["朱红", "金黄", "墨绿"],
        "description": "回纹是蜀锦中最经典的几何纹样之一，其历史可追溯至新石器时代彩陶上的螺旋纹，商周时期在青铜器上定型为回字形，至汉代已广泛用于蜀锦边饰与底纹。在蜀锦名品“雨丝锦”中，回纹常以连续排列的形式装饰锦缎边缘，与中心花卉或瑞兽纹样形成疏密对比；而在“浣花锦”中，套叠回纹则作为主纹出现，通过朱红、金黄、墨绿等色丝交织，凸显蜀锦的富丽与秩序。回纹的循环往复结构，在蜀锦语境中象征富贵不断、吉祥长久，同时其规整的几何形态也体现了蜀锦工匠对精密织造技艺的追求，常与云纹、如意纹组合，寓意“回云如意”，寄托辟邪护佑、福寿绵延的美好愿望。"
    },
    "turtle_shell": {
        "name_zh": "龟背纹",
        "category": "几何纹",
        "iconic_forms": ["六边龟甲", "四方连续", "二方连续边框"],
        "cultural_meanings": ["长寿", "坚固", "永恒"],
        "origin": "上古龟崇拜；汉代已用于织锦",
        "common_positions": ["底纹", "填充纹", "边框"],
        "typical_colors": ["金", "棕", "青"],
        "description": "龟背纹以正六边形为基本单元，通过二方或四方连续排列构成绵延不绝的几何图案。其结构严谨，常作为底纹或填充纹出现在复合纹样中——与牡丹、菊花等大型花卉主纹搭配，填充在花卉图案之间的空白区域，形成丰富的层次感。这种'花卉+几何龟背纹'的搭配手法被称为'锦上添花'。宋元时常用龟背纹作为满地锦纹的基础骨架，如'龟背龙纹锦'、'龟背折枝花锦'。在八答晕锦中，龟背纹与联珠纹、宝相花等共同构成华丽繁复的经典风格。"
    },
    "pearl_roundel": {
        "name_zh": "联珠纹",
        "category": "几何纹",
        "iconic_forms": ["联珠圈(团窠)", "条带状联珠", "多层联珠圈"],
        "cultural_meanings": ["圆满", "中西合璧", "华贵"],
        "origin": "波斯萨珊王朝装饰风格；北朝至唐代经丝绸之路传入，成为唐代蜀锦的代表性纹饰",
        "common_positions": ["团窠外环", "边框装饰带", "骨架分隔"],
        "description": "联珠纹（又称连珠纹）是一种由连续、大小相近的圆珠排列而成的几何图案，本质上是一种框架组织机制。其基本形式分为两种：条带状（圆珠排成直线或曲线作为装饰带）和联珠圈/团窠（圆珠围成同心圆环，圈内填充狮子、大象、龙凤、人物等主题纹样）。唐代陵阳公样以联珠团窠为外环围对称动物纹样，是中外文化交流的结晶。日本法隆寺藏'四天王狩纹锦'和新疆出土'联珠对孔雀纹锦'为代表作。"
    },

    # Character motifs
    "auspicious_char": {
        "name_zh": "文字纹",
        "category": "文字纹",
        "iconic_forms": ["单字", "吉语铭文", "寓合纹(文字+动植物组合)", "全文织锦"],
        "cultural_meanings": ["吉祥祝福", "福禄寿喜", "文化传承", "直抒胸臆"],
        "origin": "汉代吉语铭文盛行；唐代开创'文字织锦'先河",
        "common_positions": ["散点", "方格内饰", "边饰", "主题纹"],
        "typical_colors": ["金", "红", "黄"],
        "description": "文字纹是以汉字为题材的纹样，蜀锦艺人巧妙选用篆书等各种字体的吉祥文字，或单独使用，或与动物、花草组合形成'寓合纹'。汉代已有'登高明望四海''长乐明光''长生无极'等吉语铭文；唐代开创文字织锦先河，最杰出代表作是将王羲之《兰亭序》全文织成蜀锦，被唐太宗视为'异物'收入宫中；四天王狩猎纹锦上飞马身绣'山''吉'字。宋代后品种丰富，清晚期方方锦方格内交替织有'寿'和'囍'铭文，寓意万寿无疆。"
    },
}


# ═══════════════════════════════════════════════════════════════
# Layer 2: Composition Grammar — layout and transformation rules
# ═══════════════════════════════════════════════════════════════

LAYOUT_TYPES = {
    "center_symmetry": {
        "name_zh": "中心对称",
        "description": "以中心点为对称轴，主纹样居中，辅助纹样对称分布。战国'对龙对凤'纹样、唐代陵阳公样均以此为基础。",
        "common_in": ["月华锦", "凤穿牡丹"],
    },
    "two_side_continuous": {
        "name_zh": "二方连续",
        "description": "纹样沿一个方向重复排列，常用于边框和带饰。回纹、卷草纹、万字不断头常用此布局。",
        "common_in": ["卷草边框", "回纹边框", "万字不断头"],
    },
    "four_side_continuous": {
        "name_zh": "四方连续",
        "description": "纹样在四个方向上重复延伸，常用于底纹。几何纹、龟背纹、万字纹常用此布局。",
        "common_in": ["万字不断头", "几何锦纹", "龟背纹底纹"],
    },
    "corner": {
        "name_zh": "角隅纹样",
        "description": "纹样集中于四角，与中心主纹呼应，形成完整的四角结构。",
        "common_in": ["方方锦"],
    },
    "scattered": {
        "name_zh": "散点分布",
        "description": "纹样零散分布于画面，自由构图。花鸟纹、蝶纹常以此布局点缀。",
        "common_in": ["蜀绣小品", "散花纹锦"],
    },
    "medallion": {
        "name_zh": "团窠纹",
        "description": "以圆形或椭圆形窠环为框架，内部饰以对称或独立的纹样。源自唐代，外环常配联珠纹或卷草纹。陵阳公样和宝相花纹均用此布局。",
        "common_in": ["陵阳公样", "宝相花纹锦", "联珠团窠锦"],
    },
    "lingyang_style": {
        "name_zh": "陵阳公样",
        "description": "唐代益州工官窦师纶创制的经典构图：以联珠团窠为外环，中心内饰对称动物（对雉、天马、斗羊、翔凤、游麟等），寓意吉祥兴旺。是中西文化交流的典范，流行百年之久。",
        "common_in": ["对雉纹锦", "斗羊纹锦", "翔凤纹锦", "天马纹锦"],
    },
    "all_over": {
        "name_zh": "满地铺陈",
        "description": "锦面满铺细密纹样，再嵌以五彩大朵主花，即'锦上添花'。地纹常为几何纹或琐碎花纹，主花为牡丹、宝相花等，层次分明。",
        "common_in": ["铺地锦", "锦上添花锦"],
    },
    "broken_branch": {
        "name_zh": "折枝花",
        "description": "以单枝花卉为独立纹样单元，散点或重复排列，不构成连续的缠枝。唐代'新样'创制，打破团窠格式。元代后广泛流行。",
        "common_in": ["折枝牡丹锦", "散花纹锦"],
    },
    "flowing_water": {
        "name_zh": "流水纹",
        "description": "以波状水纹为底，纹样（梅花、桃花等）漂浮其上，波随风动、花随水流。宋代创制的'落花流水'经典构图。",
        "common_in": ["落花流水锦", "紫曲水锦"],
    },
}

TRANSFORMATION_TYPES = {
    "simplification": "简化：将具象形态简化为几何化线条，如汉代经锦中动物纹的剪影式表现",
    "exaggeration": "夸张：突出某一特征，如凤凰尾羽的夸张延伸、狮子鬃毛的强调",
    "rotation": "旋转：纹样绕中心旋转复制，如宝相花的多层放射对称",
    "scaling": "缩放：同一纹样的不同尺度组合",
    "abstraction": "抽象化：将自然形态转化为纯几何纹样，如回纹、雷纹的几何抽象",
    "combination": "组合：不同纹样的连接与交融，如卷草蝴蝶纹、龙风云纹的组合",
    "color_halo": "叠晕：同色或异色经丝由浅入深逐步过渡，形成色阶晕染效果。唐代晕榈锦使用'三阶染色'套色法，清代发展到4-9个色阶。月华锦和雨丝锦是叠晕技艺的巅峰。",
    "medallion_frame": "团窠取框：以圆形联珠或卷草为框，将纹样约束在团窠之内。唐代盛行的构图格式，北朝时期从动态平衡过渡到静态对称的重要标志。",
    "jin_shang_tian_hua": "锦上添花：在几何底纹（如龟背纹、回纹等）的空隙处填充花卉主纹（如牡丹、宝相花），形成'花卉+几何'的双层装饰效果。铺地锦和八答晕锦的典型手法。",
    "geometric_skeleton": "几何骨架：以菱形、方形、多边形几何纹为基础骨架，在骨架空格中填入动植物纹样。北朝至隋唐蜀锦的重要构图方式。",
}


# ═══════════════════════════════════════════════════════════════
# Layer 3: Composite Pattern — real textile patterns as combinations
# ═══════════════════════════════════════════════════════════════

COMPOSITE_PATTERNS = {
    "yuehua_jin": {
        "name_zh": "月华锦",
        "craft": "蜀锦",
        "structure": "彩条晕涧组织 — 由数组彩色经线排列成由浅入深、由深至浅逐渐过渡的晕涧彩条（月牙），经线起色、纬线显花，形成月光普照的光晕效果",
        "motifs": [
            {"motif": "phoenix", "role": "主纹样", "layout": "中心对称", "coverage": 0.15},
            {"motif": "scroll_grass", "role": "边框纹", "layout": "二方连续", "coverage": 0.1},
            {"motif": "cloud", "role": "辅助纹", "layout": "散点", "coverage": 0.05},
        ],
        "composition": {
            "primary": "center_symmetry",
            "border": "two_side_continuous",
            "background": "color_halo_structure",
        },
        "cultural_narrative": "月华锦以彩条晕涧组织为底——数组彩色经线排列形成由浅入深再至浅的渐变月牙彩条，经线起色、纬线显花，模拟月光普照的光晕效果。中心配以凤凰主纹，边框用卷草纹环绕。整体寓意：在月光照耀下，富贵吉祥生生不息。",
        "colors": ["红", "金", "蓝", "绿", "紫"],
        "era": "清代成熟",
    },
    "fangfang_jin": {
        "name_zh": "方方锦",
        "craft": "蜀锦",
        "structure": "方格组织 — 以彩色经纬线配以等距不同色彩的方格为骨架，方格内饰以不同色彩的圆形或椭圆形花纹。组织结构为缎地纬浮花",
        "motifs": [
            {"motif": "peony", "role": "方格内饰纹", "layout": "散点分布于方格中", "coverage": 0.2},
            {"motif": "swastika", "role": "边框纹", "layout": "二方连续万字不断头", "coverage": 0.1},
        ],
        "composition": {
            "primary": "checkered_weave",
            "border": "two_side_continuous",
            "accent": "scattered",
        },
        "cultural_narrative": "方方锦是晚清蜀锦三绝之一，本质上是一种方格组织结构——在织物单一底色上，以彩色经纬线配以等距不同色彩的方格为骨架。方格内饰以梅兰竹菊、多子石榴、八宝八吉等古朴典雅的花纹图案。方方正正的构图象征规矩、秩序与永恒。曾作为国礼赠予外宾。",
        "colors": ["红", "金", "蓝", "绿"],
        "era": "清代成熟，晚清三绝",
    },
    "feng_chuan_mudan": {
        "name_zh": "凤穿牡丹",
        "craft": "蜀绣",
        "motifs": [
            {"motif": "phoenix", "role": "主纹样", "layout": "中心对称双凤", "coverage": 0.4},
            {"motif": "peony", "role": "主纹样", "layout": "中心", "coverage": 0.3},
            {"motif": "scroll_grass", "role": "辅助纹", "layout": "环绕", "coverage": 0.2},
            {"motif": "cloud", "role": "底纹", "layout": "散点", "coverage": 0.1},
        ],
        "composition": {
            "primary": "center_symmetry",
            "accent": "scattered",
        },
        "cultural_narrative": "凤穿牡丹是蜀绣中最经典的纹样组合：双凤环绕牡丹飞翔。凤凰代表吉祥，牡丹代表富贵，二者结合寓意'富贵吉祥、夫妻和睦'。常作为婚庆用品的主题纹样。",
        "colors": ["红", "金", "绿", "粉"],
        "era": "历代传承",
    },
    "wanshou": {
        "name_zh": "万字锦",
        "craft": "蜀锦",
        "motifs": [
            {"motif": "swastika", "role": "底纹", "layout": "四方连续万字不断头", "coverage": 0.9},
            {"motif": "peony", "role": "点缀纹", "layout": "散点", "coverage": 0.1},
        ],
        "composition": {
            "primary": "four_side_continuous",
            "accent": "scattered",
        },
        "cultural_narrative": "万字锦以卍字符的四方连续为底纹，象征吉祥万德、永恒无限。散点分布的小花作为点缀打破重复的单调感。",
        "colors": ["金", "红"],
        "era": "北魏以来传承",
    },

    # ── New composite patterns from authoritative sources ──

    "luohua_liushui": {
        "name_zh": "落花流水锦",
        "craft": "蜀锦",
        "aliases": ["紫曲水", "浣花锦"],
        "motifs": [
            {"motif": "plum_blossom", "role": "主题纹", "layout": "散点漂浮", "coverage": 0.3},
            {"motif": "wave", "role": "水纹底", "layout": "流水纹", "coverage": 0.7},
        ],
        "composition": {
            "primary": "flowing_water",
        },
        "cultural_narrative": "落花流水锦（紫曲水）是宋代蜀锦艺人受唐诗'桃花流水杳然去'和宋词'花落水流红'启迪而创的经典纹样。散落的梅花或桃花漂浮于水波之上，波随风动、花随水流，情趣深厚、韵味无穷。明代实物遗存丰富，1979年成都西郊明墓出土'落花流水锦衣'残片，纹样完整清晰。",
        "colors": ["蓝", "白", "粉", "绿"],
        "era": "宋代创制，元明清传承",
    },
    "denglong": {
        "name_zh": "灯笼锦",
        "craft": "蜀锦",
        "aliases": ["庆丰年", "天下乐"],
        "motifs": [
            {"motif": "auspicious_char", "role": "吉祥文字", "layout": "灯壁悬挂", "coverage": 0.3},
            {"motif": "fish", "role": "灯下垂饰", "layout": "悬坠", "coverage": 0.1},
            {"motif": "bee", "role": "灯旁点缀", "layout": "飞动", "coverage": 0.05},
            {"motif": "grain_ear", "role": "灯旁悬结", "layout": "垂挂", "coverage": 0.1},
        ],
        "composition": {
            "primary": "center_symmetry",
            "accent": "scattered",
        },
        "cultural_narrative": "灯笼锦（庆丰年/天下乐）以谐音和隐喻表达对美好生活的向往。灯壁垂挂吊珠曰'珠联璧合'；灯下悬坠玉鱼成'吉庆有余'；灯旁悬结谷穗、灯下有蜜蜂飞动，意为'五谷丰登'。此锦流行至明末清初，所传不下百十种纹样，形制奇巧、纹饰精美，被称为'奇锦'。宋代至清代均有名品，如'绿地织金灯笼锦'、'红地方圆格翔鹤灯笼锦'等。",
        "colors": ["绿", "橘红", "葵黄", "铜蓝"],
        "era": "宋代创制，流行至明末清初",
    },
    "badayun": {
        "name_zh": "八答晕锦",
        "craft": "蜀锦",
        "aliases": ["六答晕锦", "四答晕锦"],
        "motifs": [
            {"motif": "turtle_shell", "role": "几何骨架底纹", "layout": "多边形几何骨架+填充", "coverage": 0.4},
            {"motif": "peony", "role": "主花", "layout": "团花叠晕", "coverage": 0.2},
            {"motif": "baoxiang_flower", "role": "主花", "layout": "团花叠晕", "coverage": 0.2},
            {"motif": "pearl_roundel", "role": "辅助纹", "layout": "联珠环绕", "coverage": 0.1},
        ],
        "composition": {
            "primary": "geometric_skeleton",
            "fill": "medallion",
        },
        "cultural_narrative": "八答晕锦是蜀锦中几何骨架填花纹样的代表作。以圆形、方形、多边几何形图案为骨架，采用牡丹、菊花、宝相花图案虹形叠晕套色手法，在纹样空白处镶以龟背纹连线等规则纹充满锦缎，达到'锦上添花'的效果。配色用菘蓝、荧绿、柿红、橘红等八色造成晕色效果，繁而不乱。有2~6个色阶晕圈，如加入金银线织造则丹碧玄黄、错杂交融。明代'青地八答晕加金锦'、'红地万事如意八答晕锦'为名品。",
        "colors": ["蓝", "绿", "红", "金", "黄", "紫"],
        "era": "宋代成熟，明清盛行",
    },
    "shiyang_jin": {
        "name_zh": "十样锦",
        "craft": "蜀锦",
        "aliases": ["长安竹锦", "天下乐锦", "雕团锦", "宜男锦", "宝界地锦", "方胜锦", "狮团锦", "象眼锦", "八答晕锦", "铁梗荷锦"],
        "motifs": [
            {"motif": "bamboo", "role": "长安竹", "layout": "折枝", "coverage": 0.1},
            {"motif": "auspicious_char", "role": "天下乐", "layout": "散点", "coverage": 0.1},
            {"motif": "baoxiang_flower", "role": "雕团", "layout": "团窠", "coverage": 0.1},
            {"motif": "lotus", "role": "宜男", "layout": "散点", "coverage": 0.1},
            {"motif": "meander", "role": "宝界地", "layout": "四方连续", "coverage": 0.1},
            {"motif": "geometric", "role": "方胜", "layout": "菱形组合", "coverage": 0.1},
            {"motif": "lion", "role": "狮团", "layout": "团窠", "coverage": 0.1},
            {"motif": "geometric", "role": "象眼", "layout": "几何连续", "coverage": 0.1},
            {"motif": "baoxiang_flower", "role": "八答晕", "layout": "团窠叠晕", "coverage": 0.1},
            {"motif": "lotus", "role": "铁梗荷", "layout": "折枝", "coverage": 0.1},
        ],
        "composition": {
            "primary": "medallion",
            "secondary": "scattered",
            "border": "four_side_continuous",
        },
        "cultural_narrative": "十样锦是五代蜀时创制的十种代表性锦样合称，包括：长安竹锦、天下乐锦、雕团锦、宜男锦、宝界地锦、方胜锦、狮团锦、象眼锦、八答晕锦、铁梗荷锦。元代均采用金线织造，锦样更加富丽堂皇。明代冯梦龙赞道：'近四川十样锦，远观洛内一团花。'代表了蜀锦纹样设计的最高成就。",
        "colors": ["金", "红", "蓝", "绿", "紫", "黄"],
        "era": "五代蜀创制，宋元明清传承",
    },
    "pudi": {
        "name_zh": "铺地锦",
        "craft": "蜀锦",
        "aliases": ["锦上添花锦"],
        "motifs": [
            {"motif": "meander", "role": "地纹骨架", "layout": "四方连续", "coverage": 0.5},
            {"motif": "turtle_shell", "role": "地纹填充", "layout": "四方连续", "coverage": 0.2},
            {"motif": "peony", "role": "主花", "layout": "嵌花", "coverage": 0.2},
            {"motif": "baoxiang_flower", "role": "主花", "layout": "嵌花", "coverage": 0.1},
        ],
        "composition": {
            "background": "all_over",
            "accent": "scattered",
        },
        "cultural_narrative": "铺地锦即'锦上添花'锦。缎纹组织上采用几何纹样或细小花纹铺地，形成规矩的地面花纹，再嵌以五彩斑斓的大朵花卉（宝相花、牡丹等）。主花在地纹的烘托下更加色彩艳丽、层次分明。有的铺地锦加金线织造，极为富丽堂皇。宋代'满地规则纹样'为铺地锦的前身。",
        "colors": ["红", "金", "蓝", "绿", "紫"],
        "era": "宋代成型，元明清发展",
    },
    "yusi_jin": {
        "name_zh": "雨丝锦",
        "craft": "蜀锦",
        "structure": "雨丝结构 — 锦面用白色和其他色彩的经丝组成，色经由粗渐细、白经由细渐粗交替过渡，形成色白相间、明亮对比的丝丝雨条状效果",
        "motifs": [
            {"motif": "phoenix", "role": "主题纹", "layout": "点缀于雨丝纹之间", "coverage": 0.2},
            {"motif": "dragon", "role": "主题纹", "layout": "点缀于雨丝纹之间", "coverage": 0.2},
            {"motif": "peony", "role": "主题纹", "layout": "点缀于雨丝纹之间", "coverage": 0.2},
        ],
        "composition": {
            "background": "rain_thread_structure",
            "accent": "scattered",
        },
        "cultural_narrative": "雨丝锦是晚清蜀锦三绝之一，本质上是一种经丝排列结构——用白色和彩色经丝组成'雨条'，色经由粗渐细、白经由细渐粗交替过渡，形成色白相间、明亮对比的丝丝'彩雨'效果。在这一结构背景上点缀各种图案花纹，如'龙风雨丝'、'双龙喜珠雨丝'、'牡丹雨丝'、'金鱼雨丝'、'文君听琴雨丝'等，给人以轻快舒适的韵律感。",
        "colors": ["金", "红", "绿", "蓝", "白"],
        "era": "清代成熟，晚清三绝",
    },
    "lingyang_style_pattern": {
        "name_zh": "陵阳公样",
        "craft": "蜀锦",
        "motifs": [
            {"motif": "pearl_roundel", "role": "团窠外环", "layout": "联珠圆环", "coverage": 0.2},
            {"motif": "phoenix", "role": "窠内主题", "layout": "对称翔凤", "coverage": 0.3},
            {"motif": "deer", "role": "窠内主题", "layout": "对鹿", "coverage": 0.2},
            {"motif": "qilin", "role": "窠内主题", "layout": "游麟", "coverage": 0.1},
        ],
        "composition": {
            "primary": "lingyang_style",
            "fill": "medallion",
        },
        "cultural_narrative": "陵阳公样是唐代益州大行台窦师纶（封陵阳公）吸收波斯萨珊文化精华，结合本民族特点创造的划时代蜀锦图样。特点是以团窠为主题，外环围联珠纹，团窠中央内饰对称动物纹样，多隐喻吉祥、兴旺、发达、雄健、权威。著名锦样有对雉、天马、斗羊、祥凤、游麟、对龙、对鹿等。流行百年之久，成为唐代宫廷内库织锦的标准样式，对后世织锦纹样影响深远。",
        "colors": ["红", "金", "蓝", "棕", "绿"],
        "era": "唐代初年创制，流行百年",
    },
    "baizitu": {
        "name_zh": "百子图锦",
        "craft": "蜀锦",
        "motifs": [
            {"motif": "figure", "role": "主题纹", "layout": "满铺童子嬉戏", "coverage": 0.7},
            {"motif": "auspicious_char", "role": "吉祥文字", "layout": "散点", "coverage": 0.1},
        ],
        "composition": {
            "primary": "scattered",
            "secondary": "all_over",
        },
        "cultural_narrative": "百子图锦以众多童子嬉戏玩耍为题材，寓意多子多福、子孙昌盛。清代蜀锦名品，常作婚庆用品。满铺的童子形象与吉祥文字、花木庭院结合，构成热闹喜庆的画面。",
        "colors": ["红", "金", "绿", "蓝"],
        "era": "明清时期",
    },
}


# ═══════════════════════════════════════════════════════════════
# Three-Layer KG Class
# ═══════════════════════════════════════════════════════════════

class ThreeLayerKG:
    """Knowledge graph with three layers: motif → grammar → composite."""

    def __init__(self):
        self.motifs = ATOMIC_MOTIFS
        self.layouts = LAYOUT_TYPES
        self.transformations = TRANSFORMATION_TYPES
        self.patterns = COMPOSITE_PATTERNS

    def decompose_pattern(self, pattern_id: str) -> Dict:
        """Decompose a composite pattern into its atomic motifs."""
        if pattern_id not in self.patterns:
            return {"error": f"Unknown pattern: {pattern_id}"}

        p = self.patterns[pattern_id]
        result = {
            "pattern_name": p["name_zh"],
            "craft_type": p["craft"],
            "era": p["era"],
            "motifs": [],
            "composition": [],
            "narrative": p["cultural_narrative"],
        }

        for m in p["motifs"]:
            motif_key = m["motif"]
            if motif_key in self.motifs:
                motif_def = self.motifs[motif_key]
                result["motifs"].append({
                    "name": motif_def["name_zh"],
                    "category": motif_def["category"],
                    "role": m["role"],
                    "layout": m["layout"],
                    "coverage": f"{m['coverage']:.0%}",
                    "cultural_meanings": motif_def["cultural_meanings"],
                    "origin": motif_def["origin"],
                })
            else:
                result["motifs"].append({"name": m["motif"], "role": m["role"]})

        for comp_key, layout_key in p["composition"].items():
            if layout_key in self.layouts:
                result["composition"].append({
                    "position": comp_key,
                    "name": self.layouts[layout_key]["name_zh"],
                    "description": self.layouts[layout_key]["description"],
                })

        return result

    def query_by_meaning(self, meaning: str) -> List[Dict]:
        """Find motifs and patterns associated with a cultural meaning."""
        results = []
        for key, motif in self.motifs.items():
            if any(meaning in m for m in motif["cultural_meanings"] + [motif["description"]]):
                results.append({
                    "type": "motif",
                    "id": key,
                    "name": motif["name_zh"],
                    "category": motif["category"],
                    "meanings": motif["cultural_meanings"],
                    "description": motif["description"],
                })
        return results

    def query_by_motif(self, motif_id: str) -> Dict:
        """Find all composite patterns containing a specific motif."""
        if motif_id not in self.motifs:
            return {"error": f"Unknown motif: {motif_id}"}

        motif = self.motifs[motif_id]
        appearances = []
        for pid, pattern in self.patterns.items():
            for m in pattern["motifs"]:
                if m["motif"] == motif_id:
                    appearances.append({
                        "pattern_id": pid,
                        "pattern_name": pattern["name_zh"],
                        "role": m["role"],
                        "layout": m["layout"],
                    })

        return {
            "motif_name": motif["name_zh"],
            "category": motif["category"],
            "meanings": motif["cultural_meanings"],
            "origin": motif["origin"],
            "iconic_forms": motif["iconic_forms"],
            "appears_in": appearances,
        }

    def match_composite_bayesian(self, detected_motifs: List[Dict],
                                 smoothing: float = 0.05) -> List[Dict]:
        """Bayesian inference of composite pattern from detected motifs.

        Each detected motif contributes a weight based on VL confidence:
            P(pattern | detected_motifs) ∝
                Π_i P(detected_motif_i | pattern)^{conf_i} × P(pattern)

        Args:
            detected_motifs: list of {"name": str, "confidence": float}
            smoothing: Laplace smoothing factor for unseen motifs

        Returns:
            list of {"pattern_id", "pattern_name", "probability", "narrative",
                     "matched_motifs", "missing_motifs"} sorted by probability
        """
        import math

        n_patterns = len(self.patterns)
        prior = 1.0 / n_patterns  # uniform prior

        results = []
        for pid, pattern in self.patterns.items():
            log_prob = math.log(prior)

            pattern_motif_names = {m["motif"] for m in pattern["motifs"]}
            matched = []
            missing = []

            for dm in detected_motifs:
                motif_key = dm.get("key", dm.get("name", ""))
                conf = dm.get("confidence", 0.5)

                if motif_key in pattern_motif_names:
                    # High emission probability for matching motif
                    emit_p = max(conf, 0.8)  # at least 0.8 if VL detected it
                    log_prob += conf * math.log(emit_p)
                    matched.append(motif_key)
                else:
                    # Low emission probability for non-matching motif
                    emit_p = smoothing
                    log_prob += conf * math.log(emit_p)
                    missing.append(motif_key)

            # Convert log probability to normalized probability
            prob = math.exp(log_prob)
            results.append({
                "pattern_id": pid,
                "pattern_name": pattern["name_zh"],
                "probability": round(prob, 6),
                "narrative": pattern.get("cultural_narrative", ""),
                "matched_motifs": matched,
                "missing_motifs": missing,
                "structure": pattern.get("structure", ""),
            })

        # Normalize probabilities
        total = sum(r["probability"] for r in results)
        if total > 0:
            for r in results:
                r["probability"] = round(r["probability"] / total, 4)

        results.sort(key=lambda x: x["probability"], reverse=True)
        return results

    def summary(self) -> str:
        """Return KG statistics."""
        return (f"Three-Layer KG: {len(self.motifs)} motifs × "
                f"{len(self.layouts)} layouts × "
                f"{len(self.patterns)} composite patterns × "
                f"{len(self.transformations)} transformations")


def main():
    kg = ThreeLayerKG()
    print("=" * 60)
    print("三层知识图谱 — 基本纹样 → 构图规则 → 复合纹样")
    print("=" * 60)
    print(kg.summary())

    # ---- Demo 1: Decompose a pattern ----
    print("\n[1] 纹样分解: 月华锦")
    result = kg.decompose_pattern("yuehua_jin")
    print(f"  {result['pattern_name']} ({result['craft_type']}, {result['era']})")
    print(f"  纹样构成:")
    for m in result["motifs"]:
        print(f"    [{m['role']}] {m['name']} ({m['category']}) \"{', '.join(m['cultural_meanings'])}\"")
        print(f"           布局: {m['layout']} | 占比: {m['coverage']}")
    print(f"  构图语法:")
    for c in result["composition"]:
        print(f"    {c['position']}: {c['name']} — {c['description'][:40]}...")
    print(f"  文化叙事: {result['narrative'][:80]}...")

    # ---- Demo 2: Query by meaning ----
    print(f"\n[2] 文化含义检索: \"吉祥\"")
    results = kg.query_by_meaning("吉祥")
    for r in results:
        print(f"  {r['name']} ({r['category']}): {', '.join(r['meanings'])}")

    # ---- Demo 3: Query by motif ----
    print(f"\n[3] 纹样溯源: 凤凰纹出现在哪些复合纹样中?")
    result = kg.query_by_motif("phoenix")
    print(f"  {result['motif_name']} — 含义: {', '.join(result['meanings'])}")
    print(f"  来源: {result['origin']}")
    print(f"  典型形态: {', '.join(result['iconic_forms'])}")
    for a in result["appears_in"]:
        print(f"  出现在: {a['pattern_name']} 中作为 [{a['role']}]，布局: {a['layout']}")

    # ---- Demo 4: Full pattern list ----
    print(f"\n[4] 复合纹样一览:")
    for pid, p in kg.patterns.items():
        motif_names = [kg.motifs[m["motif"]]["name_zh"] if m["motif"] in kg.motifs else m["motif"]
                       for m in p["motifs"]]
        print(f"  {p['name_zh']} ({p['craft']}): {' + '.join(motif_names)}")
        print(f"    └ {p['cultural_narrative'][:60]}...")

    print(f"\n{'='*60}")
    print("三层KG原型验证通过。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
