import os
import glob
import csv
from PIL import Image
try:
    from google import genai
except ImportError:
    print("Please install google-genai: pip install google-genai")
    exit(1)

# ──────────────────────────────────────────────
# 配置区
# ──────────────────────────────────────────────

# 存放测试图片的目录
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
# 输出的元数据文件
METADATA_FILE = os.path.join(ASSETS_DIR, "metadata.csv")

# Gemini Ground Truth 模型
GEMINI_MODEL = "gemini-3.1-pro" 
GEMINI_PROMPT = "You are an expert OCR system. Extract all text from this image as accurately as possible. Return ONLY the extracted text, without any additional formatting, markdown, or conversational text."
API_KEY = ""
# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def get_gemini_gt(client, img_path: str) -> str:
    img = Image.open(img_path)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[img, GEMINI_PROMPT]
        )
        # 清理响应中的换行符，确保 CSV 格式正确（单行文本）
        text = response.text.strip() if response.text else ""
        return text.replace("\n", " ") 
    except Exception as e:
        print(f"  [Gemini API 错误] {e}")
        return ""

def main():
    if not os.path.exists(ASSETS_DIR):
        print(f"找不到 {ASSETS_DIR} 目录，请先创建并放入图片。")
        return

    # 1. 寻找测试图片
    img_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        img_files.extend(glob.glob(os.path.join(ASSETS_DIR, ext)))

    if not img_files:
        print(f"在 {ASSETS_DIR} 目录中未找到任何图片。")
        return

    print(f"找到 {len(img_files)} 张测试图片。")

    # 2. 初始化 Gemini 客户端
    if "GEMINI_API_KEY" not in os.environ:
        print("错误: 环境变量中未找到 GEMINI_API_KEY。")
        return
    
    client = genai.Client(api_key=API_KEY)

    # 3. 生成并保存结果
    print(f"\n--- 正在生成 Ground Truth 并保存至 {METADATA_FILE} ---")
    
    # 使用 | 作为分隔符，类似 LJSpeech 格式
    with open(METADATA_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        
        for img_path in img_files:
            filename = os.path.basename(img_path)
            print(f"正在识别 {filename}...")
            gt_text = get_gemini_gt(client, img_path)
            
            if gt_text:
                writer.writerow([filename, gt_text])
                print(f"  [成功] -> {len(gt_text)} 字符")
            else:
                print(f"  [失败] 跳过 {filename}")

    print("\n完成！所有 Ground Truth 已保存。")

if __name__ == "__main__":
    main()
