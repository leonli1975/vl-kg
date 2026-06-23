"""Download Shu brocade/embroidery images from Baidu search."""

from icrawler.builtin import BaiduImageCrawler
import os

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/raw"

# ---- Core keywords ----
searches = [
    # Shu brocade
    ("shujin_general", "蜀锦 纹样"),
    ("shujin_phoenix", "蜀锦 凤凰纹"),
    ("shujin_yuehua", "蜀锦 月华"),
    ("shujin_scroll", "蜀锦 卷草纹"),
    ("shujin_cloud", "蜀锦 云纹"),
    ("shujin_fangfang", "蜀锦 方方锦"),
    ("shujin_yusi", "蜀锦 雨丝锦"),
    # Shu embroidery  
    ("shuxiu_general", "蜀绣 纹样"),
    ("shuxiu_pingxiu", "蜀绣 平绣"),
    ("shuxiu_dazi", "蜀绣 打籽绣"),
    ("shuxiu_yunzhen", "蜀绣 晕针"),
    ("shuxiu_luanzhen", "蜀绣 乱针绣"),
    ("shuxiu_phoenix", "蜀绣 凤凰"),
]

for dir_name, keyword in searches:
    save_dir = os.path.join(DATA_DIR, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading: {keyword} -> {dir_name}")
    try:
        crawler = BaiduImageCrawler(
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4,
            storage={"root_dir": save_dir},
        )
        crawler.crawl(keyword=keyword, max_num=30,
                      min_size=(200, 200),
                      file_idx_offset=0)
        count = len(os.listdir(save_dir))
        print(f"  Downloaded: {count} images")
    except Exception as e:
        print(f"  Error: {e}")

print("\nDone!")
