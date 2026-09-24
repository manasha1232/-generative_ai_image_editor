# 🎨 Text-Guided Generative AI Image Editing Studio & Inpainting Pipeline

[![Day](https://img.shields.io/badge/Day-30--30-blue?style=for-the-badge&logo=python)](https://github.com/manasha1232/30-Day-Computer-Vision-Challenge)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)

The **Grand Finale Capstone (Day 30)** of the 30-Day Computer Vision Challenge! A Deep Generative AI & Computer Vision studio for **Text-Guided Image Editing and Masked Inpainting Synthesis**. Synthesizes prompt-driven objects and textures inside target ROI masks using PyTorch latent feature representations and multi-scale Gaussian Laplacian boundary feathering.

---

## 🌟 Key Features

- 🎨 **Text-Guided Generative Editing**: Synthesizes custom objects/textures driven by free-form natural language prompts (*"futuristic glowing neon cyan sphere"*, *"cyberpunk golden artifact"*).
- 🎯 **Interactive & Programmatic ROI Mask Canvas**: Selective image region inpainting.
- 📐 **Multi-Scale Gaussian Laplacian Boundary Feathering**: Eliminates harsh seams along mask boundaries $\partial M$ ($I_{\text{blended}} = I_{\text{gen}} \cdot M_{\text{feathered}} + I_{\text{orig}} \cdot (1 - M_{\text{feathered}})$).
- 🖼️ **4-Panel Visualizer Grid**:
  1. Input Scene Image ($512 \times 512$)
  2. Target ROI Canvas Mask ($512 \times 512$)
  3. Text-Guided Generative Synthesized Output ($512 \times 512$)
  4. Spatial Blending Heatmap ($512 \times 512$)
- 📊 **Telemetry Audit Exporter**: Exports JSON telemetry detailing masked pixel coverage, boundary blending parameters, and execution performance.

---

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/manasha1232/generative_ai_image_editor.git
cd generative_ai_image_editor

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### 1️⃣ Run with Synthetic Scene & Prompt Generator
```bash
python generate_demo_editor_input.py
python generative_image_editor.py
```

### 2️⃣ Run with Custom Text Prompt
```bash
python generative_image_editor.py --prompt "cyberpunk golden artifact"
```

---

## 📊 Sample Output Telemetry JSON

```json
{
    "project": "Text-Guided Generative AI Image Editing Studio & Inpainting Pipeline",
    "day": 30,
    "status": "SUCCESS - 30-DAY CHALLENGE COMPLETED 100%",
    "resolution": {
        "width": 512,
        "height": 512
    },
    "generative_parameters": {
        "text_prompt": "futuristic glowing neon cyan sphere",
        "roi_masked_pixels": 34200,
        "masked_area_percentage": 13.05,
        "boundary_feathering_kernel": 21
    },
    "performance": {
        "execution_duration_sec": 0.412
    },
    "output_files": {
        "generative_grid_image": "output\\sample_generative_output.jpg",
        "telemetry_report": "output\\sample_editor_report.json"
    }
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
