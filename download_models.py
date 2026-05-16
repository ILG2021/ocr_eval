import os
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("请先安装 huggingface_hub: pip install huggingface_hub")
    exit(1)

# ──────────────────────────────────────────────
# 模型下载配置区
# ──────────────────────────────────────────────
# 字典映射：模型名称 -> 对应的 HuggingFace 仓库ID 及 文件列表
# 注意：由于部分模型（如 dots.ocr, qianfan-ocr 等）的开源名称和文件名可能与这里不完全一致，
# 请在实际下载前，前往 HuggingFace 搜索这些模型并修改 `repo_id` 和 `files` 为真实存在的文件名。

MODEL_DOWNLOADS = {
    "glm-ocr": {
        "repo_id": "mradermacher/GLM-OCR-GGUF",
        "files": ["GLM-OCR.Q4_K_M.gguf", "GLM-OCR.mmproj-Q8_0.gguf"],
        "save_dir": "models/glm-ocr"
    },
    "gemma4-e2b": {
        "repo_id": "unsloth/gemma-4-E2B-it-GGUF",
        "files": ["gemma-4-E2B-it-Q4_K_M.gguf", "mmproj-F16.gguf"],
        "save_dir": "models/gemma4-e2b"
    },
    "openbmb/MiniCPM-V-4.6": {
        "repo_id": "openbmb/MiniCPM-V-4.6-gguf",
        "files": ["MiniCPM-V-4_6-Q4_K_M.gguf", "mmproj-model-f16.gguf"],
        "save_dir": "models/minicpm-v-4.6"
    },
    "tencent/HunyuanOCR": {
        "repo_id": "mradermacher/HunyuanOCR-GGUF",
        "files": ["HunyuanOCR.Q4_K_M.gguf", "HunyuanOCR.mmproj-Q8_0.gguf"],
        "save_dir": "models/hunyuan-ocr"
    },
    "deepseek-ocr": {
        "repo_id": "sabafallah/DeepSeek-OCR-GGUF",
        "files": ["deepseek-ocr-Q4_K_M.gguf", "mmproj-deepseek-ocr-q8_0.gguf"],
        "save_dir": "models/deepseek-ocr"
    },
    "dots.ocr": {
        "repo_id": "mradermacher/dots.ocr-GGUF",
        "files": ["dots.ocr.Q4_K_M.gguf", "dots.ocr.mmproj-Q8_0.gguf"],
        "save_dir": "models/dots-ocr"
    },
    "qianfan-ocr": {
        "repo_id": "Reza2kn/Qianfan-OCR-GGUF",
        "files": ["Qianfan-OCR-q4_k_m.gguf", "Qianfan-OCR-mmproj-f16.gguf"],
        "save_dir": "models/qianfan-ocr"
    }
}

def main():
    print("===========================================")
    print("         OCR 模型 GGUF 权重下载脚本         ")
    print("===========================================")
    
    # 确保根级 models 目录存在
    os.makedirs("models", exist_ok=True)
    
    for model_name, info in MODEL_DOWNLOADS.items():
        repo_id = info["repo_id"]
        files = info["files"]
        save_dir = info["save_dir"]
        
        # 建立对应模型的专属文件夹
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n>>> 准备下载 [{model_name}]")
        print(f"仓库 ID: {repo_id}")
        
        for file_name in files:
            target_path = os.path.join(save_dir, file_name)
            if os.path.exists(target_path):
                print(f"  [跳过] 文件已存在: {target_path}")
                continue
                
            print(f"  正在下载: {file_name} ...")
            try:
                # 使用 hf_hub_download 下载文件，指定 local_dir 会直接保存到该目录
                downloaded_path = hf_hub_download(
                    repo_id=repo_id, 
                    filename=file_name, 
                    local_dir=save_dir,
                    local_dir_use_symlinks=False
                )
                print(f"  [成功] √ {file_name}")
            except Exception as e:
                print(f"  [失败] x 无法下载 {file_name}。原因: {e}")
                print(f"         请确认 HuggingFace 仓库 '{repo_id}' 中存在名为 '{file_name}' 的文件！")

if __name__ == "__main__":
    main()
