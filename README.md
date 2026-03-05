# Dual_Backbone_CNN_Transformer_For_SkinCancer_Classification
Hybrid CNN-Transformer skin cancer classifier achieving 87.93% accuracy with dual-backbone architecture (ConvNeXt V2 + EfficientNet V2), MSAF attention fusion, and knowledge distillation for 32× model compression. Improves melanoma detection from 21.9% to 85.8% precision on HAM10000 dataset. Production-ready with ONNX export and 120ms inference.

## 🚀 Quick Start
# Skin Cancer Classification - Setup & Execution Guide

## Prerequisites

- Google Account (for Colab)
- Kaggle Account (for dataset)

---

## Step 1: Get Kaggle API Key

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Download `kaggle.json` file

---

## Step 2: Open Notebook in Google Colab

[![Open In Colab](https://colab.research.google.com/drive/1fC2QXvEQBnzy_4yfKgtklyDJVmP0YciM?usp=sharing#scrollTo=dfv8QeuXDTbe)

---

## Step 3: Upload Kaggle Credentials

Run the first cell:

```python
from google.colab import files
uploaded = files.upload()
```

Upload your `kaggle.json` file when prompted.

---

## Step 4: Run Cells Sequentially

Execute cells in order:

### Phase 1: Data Setup 
- Installs dependencies
- Downloads HAM10000 dataset
- Creates train/val/test splits

**Runtime:** ~5 minutes

---

### Phase 2: Data Preprocessing 
- Defines augmentation pipelines
- Creates DataLoaders
- Handles class imbalance

**Runtime:** ~2 minutes

---

### Phase 3: Model Architecture 
- Loads ConvNeXt V2 & EfficientNet V2
- Builds MSAF module
- Creates Teacher model

**Runtime:** ~2 minutes

---

### Phase 4: Teacher Training 
- Trains for 20 epochs
- Saves best checkpoint

**Runtime:** ~1.5 hours (T4 GPU)

---

### Phase 5: Student Distillation 
- Trains Student model for 30 epochs
- Applies knowledge distillation

**Runtime:** ~1.5 hours (T4 GPU)

---

### Phase 6: Evaluation
- Generates metrics
- Creates confusion matrix
- Produces Grad-CAM visualizations

**Runtime:** ~5 minutes

---

### Phase 7: Export 
- Exports to ONNX format

**Runtime:** ~1 minute

---

## Expected Output Files

```
teacher_best_224.pth       (~480 MB)
student_best.pth           (~15 MB)
model_student.onnx         (~18 MB)
evaluation_results.txt
confusion_matrix.png
gradcam_samples.png
```

---

## GPU Requirements

- **Minimum:** Tesla T4 (16GB VRAM)
- **Recommended:** Tesla V100 or A100

---

## Troubleshooting

**Issue:** CUDA Out of Memory

**Solution:** Reduce batch size to 16 in Cell 6

**Issue:** Kaggle download fails

**Solution:** Verify `kaggle.json` permissions: `chmod 600 ~/.kaggle/kaggle.json`

**Issue:** Training stops at epoch 15

**Solution:** Already fixed - uses ReduceLROnPlateau scheduler

---

## Total Execution Time

**Full Pipeline:** ~3-4 hours on Colab T4 GPU

---


## 📧 Contact

- **Author:** Chetan Kumar Patruni
- **LinkedIn:** https://www.linkedin.com/in/chetan-kumar-patruni/

## 🙏 Acknowledgments

- Base paper: Jang & Park (2024) - DER Algorithm
- HAM10000 dataset: Tschandl et al. (2018)
- TIMM library maintainers
