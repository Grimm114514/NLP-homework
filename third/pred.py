# -*- coding: gbk -*-
import os
from openai import OpenAI

# 1. 配置
INPUT_FILE = './corpus_use/50_lines_test.txt'
OUTPUT_FILE = './corpus_pred/segmented_data.txt'
MODEL_NAME = "Qwen3-8B"
BATCH_SIZE = 20  # 一次发给模型处理的行数 (建议 10-50 之间，取决于句子长度)

# 2. 读取 Key
try:
    with open('api.txt', 'r') as f:
        api_key = f.read().strip()
except FileNotFoundError:
    print("错误：未找到 api.txt")
    exit()

client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key=api_key,
    default_headers={"X-Failover-Enabled": "true"},
)

def process_batch(lines_batch):
    """
    将一批句子合并发送，并要求模型保持多行格式返回
    """
    # 将列表合并成一个长字符串
    input_text = "\n".join(lines_batch)
    
    # 针对多行处理优化的 Prompt
    system_prompt = (
        "You are a high-efficiency Chinese Word Segmentation tool.\n"
        "Task: Segment the provided text into words using spaces.\n"
        "Strict Constraints:\n"
        "1. The input contains multiple lines. You must process ALL lines.\n"
        "2. Output format must strictly match the line count of the input.\n"
        "3. Do NOT merge lines. Do NOT delete lines.\n"
        "4. Only output the segmented text. No header, no footer, no markdown.\n"
        "5. Use single spaces for segmentation."
    )

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ],
            model="DeepSeek-V3",
            stream=False, 
            max_tokens=4096,
            temperature=0.1,
        )
        
        # 获取结果并去除首尾空白
        result = response.choices[0].message.content.strip()
        
        # 如果模型输出了 markdown 代码块 (```), 尝试去除
        if result.startswith("```"):
            result = result.replace("```text", "").replace("```", "").strip()
            
        return result

    except Exception as e:
        print(f"\n[Batch Error]: {e}")
        return "\n".join(lines_batch) # 出错时返还原门，防止丢数据

def main():
    # 清空输出文件
    with open(OUTPUT_FILE, 'w', encoding='gbk', errors='ignore') as f:
        pass

    if not os.path.exists(INPUT_FILE):
        print(f"找不到文件: {INPUT_FILE}")
        return

    # --- 1. 读取数据 (GBK 兼容) ---
    print(f"正在读取 {INPUT_FILE} ...")
    lines = []
    
    with open(INPUT_FILE, 'r', encoding='gbk') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    total_lines = len(lines)
    print(f"共 {total_lines} 行有效数据。每批处理 {BATCH_SIZE} 行。")

    # --- 2. 批量循环处理 ---
    with open(OUTPUT_FILE, 'a', encoding='gbk', errors='ignore') as f_out:
        for i in range(0, total_lines, BATCH_SIZE):
            # 取出一个批次
            batch = lines[i : i + BATCH_SIZE]
            
            print(f"正在处理第 {i+1} - {min(i+BATCH_SIZE, total_lines)} 行...", end="", flush=True)
            
            # 调用 API
            segmented_block = process_batch(batch)
            
            # 写入结果 (自动换行)
            if segmented_block:
                f_out.write(segmented_block + "\n")
            
            print(" 完成")

    print(f"\n处理完毕！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()