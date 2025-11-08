import re
import os

def clean_text_for_nlp(text):
    """
    一个专门为NLP分析（如词频、Zipf定律）定制的文本清理函数。
    它会智能处理空格，以正确分离英文单词，同时保留段落结构。
    """
    
    # 1. 移除特定的爬虫/系统残留信息
    text = re.sub(r'\\', '', text)
    text = re.sub(r'Flash Player插件.*?(确认取消)', '', text, flags=re.DOTALL)
    text = re.sub(r'炒股就看.*主题机会！', '', text)
    text = re.sub(r'【本文结束】.*', '', text, flags=re.MULTILINE)
    text = re.sub(r'声明：市场有风险.*', '', text, flags=re.DOTALL)

    # 2. 移除URL和电子邮件
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    # 3. 移除来源、作者、编辑等归属信息
    text = re.sub(r'^(来源：|作者：|原标题：|转载自：|出品：|编辑整理：|责任编辑：|媒体来源：|【来源：|（来源：|（总台央视记者|（作者为|（国际锐评评论员）|（新浪母婴研究院专家).*$', 
                  '', text, flags=re.MULTILINE)
    text = re.sub(r'[（\(][^）\)]*?(责编|记者|编辑|作者|来源|编译|图/|摄影|/文|实习生|评论员|据|ETtoday|AIGC|...)[^）\)]*?[）\)]', 
                  '', text)
    text = re.sub(r'[（\(][\u4e00-\u9fa5]{1,5}[）\)]', '', text)
    text = re.sub(r'[（\(][A-Za-z\s]{1,20}[）\)]', '', text)
    text = re.sub(r'(\w+记者)\s[\u4e00-\u9fa5\s]+', '', text)
    text = re.sub(r'\b[A-Z]{1,2}\d{3,}\b', '', text)
    
    # 4. 移除股票代码、日期和新闻社样板信息
    text = re.sub(r'[（\(]\d{5,6}\.(SH|SZ|HK)[^）\)]*[）\)]', '', text)
    text = re.sub(r'(北京时间)?(\d{4}年)?\d{1,2}月\d{1,2}日', '', text)
    text = re.sub(r'IT之家 \d{1,2}月 \d{1,2} 日消息', '', text)
    text = re.sub(r'（中新网.*?电）', '', text)
    text = re.sub(r'\《.*?\》.*?文章，原题：', '', text)

    # 5. 移除特殊符号
    text = re.sub(r'[●▲◎□★]', '', text)

    # 6. (核心步骤) 将所有非 汉字/字母/数字/换行符 的字符替换为空格
    # 这会保留 \n (换行符)
    # 并且把所有标点符号变成空格，这对分离英文单词至关重要
    text = re.sub(r'[^\w\s\u4e00-\u9fa5\n]', ' ', text, flags=re.UNICODE)
    
    # 7. (新) 规范化水平空白
    # 将所有水平空格（普通空格, tab, 全角空格, 下划线）压缩为单个空格
    text = re.sub(r'[ \t\u3000_]+', ' ', text)
    
    # 8. (新) 规范化垂直空白
    # 将多个换行符（空行）压缩为单个换行符
    text = re.sub(r'\n+', '\n', text)
    
    # 9. (新) 移除空行和行首尾的空格
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    return '\n'.join(cleaned_lines)

def main():
    input_filename = 'sina_scraper/150cn.txt'
    output_filename = 'sina_scraper/150cn_cleaned.txt' # 换个名字

    # 确保输入文件存在
    if not os.path.exists(input_filename):
        print(f"错误: 输入文件 '{input_filename}' 未找到。")
        return

    # --- 开始处理 ---
    print(f"正在读取文件: {input_filename}")
    with open(input_filename, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 执行清理
    print("正在清理文本...")
    cleaned_content = clean_text_for_nlp(content)
    
    # 保存清理后的文件
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
        
    print(f"\n--- 清理完成 ---")
    print(f"原始文件: {input_filename}")
    print(f"清理后文件: {output_filename}")
    
    # 预览清理后的文本
    print("\n--- 清理后文本预览 (前10行) ---")
    if cleaned_content:
        print('\n'.join(cleaned_content.split('\n')[:10]))
    else:
        print("（清理后内容为空）")
    print("---------------------------------")

if __name__ == "__main__":
    main()