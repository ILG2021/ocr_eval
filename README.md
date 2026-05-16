# OCR Eval: 多模型 OCR 性能评测工具

这是一个基于 `llama.cpp` (llama-server) 的多模型 OCR 评测工具。它可以自动下载多个主流开源多模态大模型（GGUF 格式），并针对指定的图片集进行识别准确率、推理耗时以及显存占用的全方位评测。

## 🌟 核心功能

- **多模型支持**：支持 GLM-OCR, Gemma 4V, MiniCPM-V, Hunyuan-OCR, DeepSeek-OCR 等多种 GGUF 视觉模型。
- **自动环境管理**：集成了模型下载、`llama-server` 服务的自动启动、关闭与无缝切换。
- **高精度评测指标**：
  - **准确率 (Accuracy)**：使用 `difflib` 计算识别文本与标准答案（Ground Truth）的相似度。
  - **推理耗时 (Latency)**：统计单张图片的端到端识别时间。
  - **峰值显存 (Peak VRAM)**：实时监控 GPU 显存，自动减去系统初始占用，计算模型运行的净峰值显存。
- **Ground Truth 自动生成**：支持调用 Google Gemini API 自动为测试图片生成高质量的标准答案。

## 📁 目录结构

```text
ocr_eval/
├── assets/                 # 测试图片存放目录
│   └── metadata.csv        # 图片名与标准答案 (filename|text)
├── models/                 # 模型权重存放目录 (自动生成)
├── tools/                  # 存放 llama-server 可执行文件
├── download_models.py      # 模型自动下载脚本
├── generate_gt.py          # Ground Truth 自动生成脚本 (使用 Gemini)
├── evaluate_ocr.py         # 核心评测脚本
├── llama_server_manager.py # llama-server 进程管理器
└── evaluation_results.txt  # 评测结果保存记录
```

## 🚀 快速开始

### 1. 安装依赖
确保你已经安装了 Python 3.x，并安装相关依赖：
```bash
pip install requests pillow google-generativeai huggingface_hub
```

### 2. 准备图片与标准答案
- 将你的测试图片放入 `assets/` 文件夹。
- **方式 A (手动)**：在 `assets/metadata.csv` 中按 `文件名|标准文字内容` 格式填写。
- **方式 B (自动)**：
  1. 在 `generate_gt.py` 中填入你的 `API_KEY` (Gemini)。
  2. 运行 `python generate_gt.py` 自动生成标准答案。

### 3. 下载模型
运行以下脚本，自动从 Hugging Face 下载所需的 GGUF 模型文件：
```bash
python download_models.py
```

### 4. 运行评测
执行主评测脚本，系统将依次启动各个模型并输出报告：
```bash
python evaluate_ocr.py
```

## 📊 评测结果示例

脚本运行完成后，会在控制台输出汇总表格，并追加保存至 `evaluation_results.txt`：

## 🛠️ 注意事项

- **显存要求**：建议使用具有 8GB 以上显存的 NVIDIA 显卡。
- **Context Size**：默认为每个模型分配了 4096~8192 的上下文长度，以支持高分辨率图片处理。如果遇到 400 错误，请检查 `llama_server_manager.py` 中的 `ctx_size` 设置。
- **llama-server**：请确保 `tools/` 目录下有对应操作系统的 `llama-server` 可执行文件。

## ⚖️ 许可证
本项目采用 MIT 许可证。
