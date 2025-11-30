# 3SDeblur: Three-Stage Image Defocus Deblurring Guided by Residual Prior

[![Paper](https://img.shields.io/badge/Paper-Springer-brightgreen)](https://link.springer.com/chapter/10.1007/978-981-95-4378-6_6)
[![Code](https://img.shields.io/badge/Code-Github-blue)](https://github.com/lizhangray/3SDeblur)
[![Dataset](https://img.shields.io/badge/Dataset-LFDOF-orange)](https://drive.google.com/drive/folders/1whycpa0nLo4zVUg6s9h5Di9ZkOF5Ua7i?usp=sharing)

Official PyTorch implementation of the paper:  
**"3SDeblur: Three-Stage Image Defocus Deblurring Guided by Residual Prior"**  
*Yihang Chen, Zhan Li, Boyang Yao, Xiaohan Li, Wenzhuo Wang, Qingliang Chen*  
Department of Computer Science, Jinan University

---

## 📌 Task

**Single Image Defocus Deblurring (SIDD)** aims to restore a sharp all-in-focus image from a blurry input degraded by defocus blur. This is a challenging ill-posed problem due to information loss and non-uniqueness of solutions.

---

## 📖 Abstract

As a typical low-level vision task, single image defocus deblurring aims to produce a sharp clear image from a defocused blurry image. However, image degradation inevitably results in a significant loss of information and non-uniqueness of the solution, making it challenging to learn the mapping from low-quality inputs to high-quality outputs. To address this issue, we propose a three-stage learning framework for image defocus deblurring, namely 3SDeblur, which consists of stages for accurate feature extraction, residual prior estimation, and image restoration. First, the solution space is constrained by applying identity mapping to all-in-focus images, to extract an accurate feature representation (AFR) of ground truth images. Second, a dynamic fractional derivative network is designed to estimate residual maps as a prior to compensate for the lost details caused by the defocus blurring. Finally, we propose a residual-guided attention module to aggregate the defocused image and residual map. Under the guidance of AFR and residual priors, a sharp image is restored. Extensive experiments demonstrate that our 3SDeblur achieves superior performance on defocus deblurring, especially a significant improvement of 0.81 dB in PSNR on the LFDOF dataset, compared to state-of-the-art methods.

---

## 🎯 Results

### Qualitative Comparison

[Qualitative Results](https://drive.google.com/uc?export=view&id=YOUR_IMAGE_ID_HERE)


### Quantitative Comparison on LFDOF Dataset

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | 
|--------|--------------|---------------|---------------|
| NAFNet*  | 30.52 | 0.882 | 0.163 |
| Restormer*  | 29.83 | 0.880 | 0.151 |
| INIKNet | 30.29 | 0.886 | 0.132 |
| NRKNet  | 30.48 | 0.884 | 0.147 |
| ViTDeblur  | 30.52 | 0.892 | 0.144 |
| DocRes*  | 29.86 | 0.873 | 0.163 |
| **3SDeblur** | **31.33** | **0.896** | **0.127** |

### Quantitative Comparison on DPDD Dataset

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Params(M) | FLOPs(G) | Time(s) |
|--------|-------|-------|--------|-----------|----------|---------|
| DRBNet | 25.72 | 0.803 | 0.185 | 11.69 | 49.2 | 1.16 |
| NAFNet* | 25.39 | 0.846 | 0.199 | 17.06 | 16.0 | 0.56 |
| Restormer* | 25.98 | 0.811 | 0.178 | 26.10 | 141.0 | 1.08 |
| INIKNet | 26.06 | 0.803 | 0.185 | 1.98 | 79.3 | 1.56 |
| NRKNet | 26.11 | 0.810 | 0.210 | 6.09 | 78.6 | 1.30 |
| ViTDeblur | 26.11 | 0.814 | 0.201 | 113.33 | 111.8 | 0.94 |
| DocRes* | 25.61 | 0.796 | 0.246 | 15.19 | 91.9 | 0.72 |
| **3SDeblur** | **26.09** | **0.866** | **0.131** | **17.66** | **69.1** | **0.71** |


---

## 🛠 Installation

```bash
git clone https://github.com/lizhangray/3SDeblur.git
cd 3SDeblur
```