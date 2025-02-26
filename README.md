# Fine-Tuning LLaMA 2-7B with Dolly 15K

## 📌 Project Overview
This project fine-tunes **Meta LLaMA 2-7B-Chat** using the **Databricks Dolly 15K** dataset. The goal is to adapt the model for **instruction-based conversation generation**, leveraging **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized LoRA)** to make the fine-tuning process more efficient.

## ✨ Features
- Utilizes **LoRA** for memory-efficient fine-tuning.
- Uses **QLoRA** for 4-bit quantization, reducing VRAM requirements.
- Implements **Hugging Face's `trl` (Transformer Reinforcement Learning)** library for training.
- Stores the fine-tuned model using **PEFT (Parameter Efficient Fine-Tuning)** for efficient loading and inference.

## 📦 Installation
Clone this repository and install the necessary dependencies:

```bash
pip install -qU datasets trl bitsandbytes accelerate peft transformers
```

## 🏗 Model and Dataset
- **Model:** `meta-llama/Llama-2-7b-chat-hf`
- **Dataset:** `databricks/databricks-dolly-15k`
  - Contains 15,000 high-quality, human-written instruction-following records.
  
## 🚀 Training Pipeline
The fine-tuning process follows these steps:
1. **Load the dataset** from Hugging Face (`databricks/databricks-dolly-15k`).
2. **Preprocess the dataset** by structuring prompts in a chat-like format.
3. **Apply quantization (QLoRA)** to optimize memory usage.
4. **Fine-tune the model** using LoRA adapters.
5. **Save and merge the LoRA weights** for model inference.

## 📂 Repository Structure
```bash
llama2-finetune/
│
├── fine_tuning_llama2.ipynb
├── data/
│   ├── llama2_qa.csv    # Preprocessed csv
│   └── llama2_qa_ds     # Preprocessed dataset
│
├── model/               # fine-tuned model
│   └── checkpoint-500   # model files
└── README.md
```
