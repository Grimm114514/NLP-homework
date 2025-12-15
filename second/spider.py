import requests
from bs4 import BeautifulSoup
import time
import os
import re

# === 配置区域 ===
# 目标语言: 'en' (英语) 或 'zh' (中文)
# 建议选 'en'，因为你的 data.py 目前只支持英语分词
LANG = 'en' 

# 目标文件大小 (单位: MB)
# 作业要求不高，爬 2MB - 5MB 纯文本就足够训练出不错的效果了
TARGET_SIZE_MB = 2 

# 输出文件名
OUTPUT_FILE = 'corpus.txt'

# =================

def get_random_article_url(lang):
    """获取维基百科随机条目的URL"""
    return f"https://{lang}.wikipedia.org/wiki/Special:Random"

def clean_text(text):
    """简单的文本清洗"""
    # 去除引用标记，如 [1], [2]
    text = re.sub(r'\[\d+\]', '', text)
    # 去除多余的空白
    text = text.strip()
    return text

def crawl_wikipedia():
    print(f"🕷️ 开始爬取维基百科 ({LANG})，目标大小: {TARGET_SIZE_MB} MB...")
    print(f"📂 结果将保存至: {OUTPUT_FILE}\n")

    # 如果文件已存在，先清空
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("")

    current_size = 0
    article_count = 0
    target_bytes = TARGET_SIZE_MB * 1024 * 1024

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    while current_size < target_bytes:
        try:
            url = get_random_article_url(LANG)
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 提取标题
                title = soup.find('h1', {'id': 'firstHeading'}).text
                
                # 提取正文 (维基百科的正文都在 <p> 标签里)
                paragraphs = soup.find_all('p')
                content = []
                for p in paragraphs:
                    text = clean_text(p.get_text())
                    if len(text) > 50: # 忽略太短的段落
                        content.append(text)
                
                full_text = " ".join(content) + "\n"
                
                # 只有当提取到有效内容时才写入
                if len(full_text) > 200:
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                        f.write(full_text)
                    
                    # 更新统计
                    file_size = os.path.getsize(OUTPUT_FILE)
                    current_size = file_size
                    article_count += 1
                    
                    print(f"[{article_count}] 已爬取: {title[:20]:<20}... | 当前大小: {file_size/1024:.2f} KB")
                
            else:
                print(f"⚠️ 请求失败: {response.status_code}")

            # 礼貌爬虫，防止被封 IP
            time.sleep(1.0)

        except Exception as e:
            print(f"❌ 发生错误: {e}")
            time.sleep(2)

    print(f"\n✅ 爬取完成！总共爬取了 {article_count} 篇文章。")
    print(f"📄 文件已保存为 {OUTPUT_FILE}，大小: {current_size/1024/1024:.2f} MB")
    print("🚀 现在你可以运行 main.py 了！")

if __name__ == "__main__":
    # 你需要安装 requests 和 beautifulsoup4
    # pip install requests beautifulsoup4
    crawl_wikipedia()