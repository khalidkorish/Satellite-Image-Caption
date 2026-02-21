# 🛰️ Satellite Image Caption — EfficientNet-B3 + Attention LSTM

> Automatically generate natural-language descriptions for satellite and aerial images using a deep learning pipeline trained on the RSICD dataset.

---

## 📌 Project Overview

This project builds an end-to-end **image captioning system** designed specifically for **remote sensing / satellite imagery**. Given a satellite image, the model outputs a single, human-readable sentence describing the scene.

**Example:**
> *"A residential area with many buildings surrounded by green trees."*

---

## 🏗️ Architecture

```
Satellite Image (224×224)
        │
        ▼
┌─────────────────────┐
│  EfficientNet-B3    │  ← Pretrained CNN Encoder (timm)
│  Encoder            │     Output: (B, 49, 1536) — 7×7 spatial grid
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Bahdanau Attention │  ← Learns WHERE to look in the image
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  LSTM Decoder       │  ← Generates caption word by word
└─────────────────────┘
        │
        ▼
   Generated Caption
```

---

## 📂 Dataset — RSICD

| Property | Value |
|---|---|
| Total images | ~10,921 |
| Image size | 224 × 224 px |
| Captions per image | 5 |
| Source | Google Earth, Baidu Map |
| Format | CSV with raw image bytes |

The **Remote Sensing Image Caption Dataset (RSICD)** covers a wide range of scene categories including airports, beaches, forests, farmland, residential areas, and more.

📥 **Dataset:** [RSICD on Kaggle](https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset)

---

## ⚙️ Training Details

| Hyperparameter | Value |
|---|---|
| Encoder | EfficientNet-B3 (pretrained, timm) |
| Decoder | LSTM with Bahdanau Attention |
| Optimizer | AdamW |
| LR (decoder) | 4e-4 |
| LR (encoder) | 1e-4 (fine-tuned after warm-up) |
| Scheduler | CosineAnnealingLR |
| Label Smoothing | 0.1 |
| Max training samples | 6,000 |
| Inference | Beam Search |

---

## 🚀 Gradio Demo

The project includes an interactive **Gradio web demo** where you can upload any satellite image and get an instant caption.

> 📹 **See `Project_Demo.mp4`** in this repo for a full walkthrough of the demo.

---

## 📁 Repository Structure

```
├── rsicd-efficientnet-lstm.ipynb   # Main notebook (training + demo)
├── Project_Demo.mp4                # Gradio demo screen recording
└── README.md                       # This file
```

---

## 🛠️ Requirements

```bash
pip install timm gradio torch torchvision pandas numpy pillow plotly kaleido
```

---

## 📖 How to Use

1. **Clone the repo**
   ```bash
   git clone https://github.com/khalidkorish/Satellite-Image-Caption.git
   cd Satellite-Image-Caption
   ```

2. **Open the notebook** on Kaggle or locally (GPU recommended)

3. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset) and update `cfg.BASE_DIR` in the notebook

4. **Run all cells** — the last cell launches the Gradio demo automatically

5. **Upload a satellite image** → get your caption!

---

## 📊 Results

The model is evaluated using **BLEU score** on the RSICD test split. Training and validation loss curves are visualized interactively using Plotly inside the notebook.

---

## 🙏 Acknowledgements

- [RSICD Dataset on Kaggle](https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset)
- [timm — PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [Gradio](https://gradio.app/)

---

## 👤 Author

**Khalid Korish**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Khalid%20Korish-blue?logo=linkedin)](https://www.linkedin.com/in/khalidkorish/)
[![GitHub](https://img.shields.io/badge/GitHub-khalidkorish-black?logo=github)](https://github.com/khalidkorish/Satellite-Image-Caption)

---

## 📄 License

This project is for educational and research purposes.
