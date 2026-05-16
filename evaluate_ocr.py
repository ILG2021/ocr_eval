import os
import glob
import base64
import difflib
import requests
import time
import json
from pathlib import Path
from PIL import Image
import csv
import threading
import subprocess
import llama_server_manager

# ──────────────────────────────────────────────
# 配置区
# ──────────────────────────────────────────────

# 存放测试图片的目录
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
# 元数据文件 (包含 Ground Truth)
METADATA_FILE = os.path.join(ASSETS_DIR, "metadata.csv")

# 模型配置字典 (映射模型名到其 gguf 文件和 mmproj 文件路径)
# 需要您根据实际下载的本地文件路径进行调整
MODELS_CONFIG = {
    "glm-ocr": {
        "model_path": "models/glm-ocr/GLM-OCR.Q4_K_M.gguf",
        "mmproj_path": "models/glm-ocr/GLM-OCR.mmproj-Q8_0.gguf",
        "prompt": "Please extract the text in the image.",
    },
    "gemma4-e2b": {
        "model_path": "models/gemma4-e2b/gemma-4-E2B-it-Q4_K_M.gguf",
        "mmproj_path": "models/gemma4-e2b/mmproj-F16.gguf",  # 如果该模型不需要mmproj则填 None
        "prompt": "Extract the text from the image.",
    },
    "openbmb/MiniCPM-V-4.6": {
        "model_path": "models/minicpm-v-4.6/MiniCPM-V-4_6-Q4_K_M.gguf",
        "mmproj_path": "models/minicpm-v-4.6/mmproj-model-f16.gguf",
        "prompt": "OCR",
    },
    "tencent/HunyuanOCR": {
        "model_path": "models/hunyuan-ocr/HunyuanOCR.Q4_K_M.gguf",
        "mmproj_path": "models/hunyuan-ocr/HunyuanOCR.mmproj-Q8_0.gguf",
        "prompt": "识别图中的所有文字。",
    },
    "deepseek-ocr": {
        "model_path": "models/deepseek-ocr/deepseek-ocr-Q4_K_M.gguf",
        "mmproj_path": "models/deepseek-ocr/mmproj-deepseek-ocr-q8_0.gguf",
        "prompt": "Extract the text from the image.",
    },
    "dots.ocr": {
        "model_path": "models/dots-ocr/dots.ocr.Q4_K_M.gguf",
        "mmproj_path": "models/dots-ocr/dots.ocr.mmproj-Q8_0.gguf",
        "prompt": "Please read the text in this image.",
    },
    "qianfan-ocr": {
        "model_path": "models/qianfan-ocr/Qianfan-OCR-q4_k_m.gguf",
        "mmproj_path": "models/qianfan-ocr/Qianfan-OCR-mmproj-f16.gguf",
        "prompt": "提取图片中的文字。",
    }
}

LLAMA_SERVER_URL = "http://127.0.0.1:1241/v1/chat/completions"


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

class VRAMMonitor:
    def __init__(self, baseline_vram=0.0):
        self.is_running = False
        self.peak_vram = 0.0
        self.baseline_vram = baseline_vram
        self.thread = None

    def _monitor(self):
        while self.is_running:
            try:
                kwargs = {}
                if os.name == 'nt':
                    kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    **kwargs
                ).decode('utf-8')
                vram_used = sum(float(x.strip()) for x in output.strip().split('\n') if x.strip())
                net_vram = vram_used - self.baseline_vram
                if net_vram > self.peak_vram:
                    self.peak_vram = net_vram
            except Exception:
                pass
            time.sleep(0.5)

    def start(self):
        self.is_running = True
        self.peak_vram = 0.0
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        return self.peak_vram


def image_to_base64(img_path: str) -> str:
    with open(img_path, "rb") as f:
        img_data = f.read()
    b64_data = base64.b64encode(img_data).decode('utf-8')
    ext = os.path.splitext(img_path)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64_data}"


def calculate_accuracy(gt: str, pred: str) -> float:
    """使用 difflib 计算字符串相似度作为准确率 (类似 1 - CER)"""
    if not gt and not pred:
        return 1.0
    return difflib.SequenceMatcher(None, gt.strip(), pred.strip()).ratio()


def load_ground_truth() -> dict:
    """从 metadata.csv 加载 Ground Truth"""
    gt_results = {}
    if not os.path.exists(METADATA_FILE):
        print(f"警告: 找不到元数据文件 {METADATA_FILE}。请先运行 generate_gt.py")
        return gt_results

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) >= 2:
                filename, text = row[0], row[1]
                img_path = os.path.join(ASSETS_DIR, filename)
                gt_results[img_path] = text
    return gt_results


def test_model_on_image(img_path: str, prompt: str) -> str:
    """发送图片给本地 llama-server"""
    b64_img = image_to_base64(img_path)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": b64_img
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,  # 评测通常需要确定性输出
    }
    try:
        t0 = time.time()
        r = requests.post(LLAMA_SERVER_URL, json=payload, timeout=60)
        r.raise_for_status()
        t1 = time.time()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip(), t1 - t0
    except Exception as e:
        print(f"  [llama-server 请求错误] {e}")
        return "", 0.0


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    # 1. 寻找测试图片
    if not os.path.exists(ASSETS_DIR):
        print(f"找不到 {ASSETS_DIR} 目录，请先创建并放入图片。")
        return

    img_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        img_files.extend(glob.glob(os.path.join(ASSETS_DIR, ext)))

    if not img_files:
        print(f"在 {ASSETS_DIR} 目录中未找到任何图片。")
        return

    print(f"找到 {len(img_files)} 张测试图片。")

    # 2. 加载 Ground Truth (GT)
    print("\n--- [阶段 1] 加载 Ground Truth (metadata.csv) ---")
    gt_results = load_ground_truth()

    # 过滤出有 GT 的图片进行测试
    img_files = [img for img in img_files if img in gt_results]

    if not img_files:
        print("没有可供评估的图片（缺失 Ground Truth）。")
        return

    print(f"准备评估 {len(img_files)} 张图片。")

    # 3. 逐个模型运行并评估
    print("\n--- [阶段 2] 使用各个本地模型识别并评估 ---")

    results_summary = {}

    # 获取一次纯净的系统初始显存占用
    llama_server_manager.stop_server()
    time.sleep(1)  # 稍作等待以确保进程和显存完全释放
    try:
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            **kwargs
        ).decode('utf-8')
        system_baseline_vram = sum(float(x.strip()) for x in output.strip().split('\n') if x.strip())
        print(f"系统初始显存占用: {system_baseline_vram:.0f} MB")
    except Exception:
        system_baseline_vram = 0.0

    for model_name, cfg in MODELS_CONFIG.items():
        print(f"\n[{model_name}] 准备启动服务...")

        # 检查配置中的文件是否存在
        model_abs_path = os.path.join(os.path.dirname(__file__), cfg["model_path"])
        if not os.path.exists(model_abs_path):
            print(f"  跳过 {model_name}，因为找不到模型文件: {model_abs_path}")
            continue

        mmproj_abs_path = None
        if cfg.get("mmproj_path"):
            mmproj_abs_path = os.path.join(os.path.dirname(__file__), cfg["mmproj_path"])

        # 启动 llama-server
        success = llama_server_manager.switch_model(
            model_path=model_abs_path,
            mmproj_path=mmproj_abs_path
        )

        if not success:
            print(f"  启动 {model_name} 失败，跳过。")
            continue

        print(f"  服务就绪，开始测试 {len(img_files)} 张图片...")

        vram_monitor = VRAMMonitor(baseline_vram=system_baseline_vram)
        vram_monitor.start()

        model_scores = []
        model_times = []
        for img in img_files:
            gt_text = gt_results.get(img, "")
            pred_text, latency = test_model_on_image(img, cfg["prompt"])

            acc = calculate_accuracy(gt_text, pred_text)
            model_scores.append(acc)
            model_times.append(latency)
            print(f"  - {os.path.basename(img)}: 准确率 {acc:.2%} | 耗时 {latency:.2f}s")

        peak_vram = vram_monitor.stop()

        avg_acc = sum(model_scores) / len(model_scores) if model_scores else 0
        avg_time = sum(model_times) / len(model_times) if model_times else 0
        results_summary[model_name] = {
            "avg_acc": avg_acc,
            "avg_time": avg_time,
            "peak_vram": peak_vram
        }
        print(f"[{model_name}] 平均准确率: {avg_acc:.2%} | 平均耗时: {avg_time:.2f}s | 峰值显存: {peak_vram:.0f} MB")

    # 4. 关闭 llama-server
    llama_server_manager.stop_server()

    # 5. 输出最终报告
    print("\n=======================================================================")
    print("                           OCR 评测最终结果                            ")
    print("=======================================================================")
    # 按照准确率排序
    sorted_results = sorted(results_summary.items(), key=lambda x: x[1]["avg_acc"], reverse=True)
    if not sorted_results:
        print("没有成功测试任何模型。")
    else:
        for i, (name, stats) in enumerate(sorted_results, 1):
            print(
                f"{i}. {name:<25} 准确率: {stats['avg_acc']:.2%} | 平均耗时: {stats['avg_time']:.2f}s | 峰值显存: {stats['peak_vram']:.0f} MB")
    print("=======================================================================")
    print("评测完成！")

    # 6. 追加保存到文件中
    result_file = os.path.join(os.path.dirname(__file__), "evaluation_results.txt")
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- 评测时间: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        if not sorted_results:
            f.write("没有成功测试任何模型。\n")
        else:
            for i, (name, stats) in enumerate(sorted_results, 1):
                f.write(
                    f"{i}. {name:<25} 准确率: {stats['avg_acc']:.2%} | 平均耗时: {stats['avg_time']:.2f}s | 峰值显存: {stats['peak_vram']:.0f} MB\n")
        f.write("-" * 50 + "\n")
    print(f"结果已追加保存至 {result_file}")


if __name__ == "__main__":
    main()
