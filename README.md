# S²A-Net: Spatially Self-Aware Multitask Network for Joint Retinal Structure Segmentation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-orange)](https://pytorch.org/)

Official implementation of **S²A-Net**, a unified CNN-Transformer framework for joint segmentation of retinal vessels (RV) and foveal avascular zone (FAZ) in OCTA images. Achieves state-of-the-art performance through spatial self-awareness and dynamic task balancing.

![S2A-Net Architecture](docs/network.png)  
*Figure: Overall architecture of S²A-Net (Illustration suggested)*

## 📖 Introduction
Accurate segmentation of retinal structures is crucial for ophthalmic biomarker analysis. Existing methods either:
- ❌ **Independently** segment RV/FAZ (ignoring anatomical relationships)
- ❌ Use **rigid joint-learning** (failing to balance task competition)

**Our Solution**:  
**S²A-Net** introduces:
1. **Spatial Self-Awareness**: Unified Transformer-CNN framework modeling retinal structure relationships
2. **Dynamic Weight Balancing**: Sinusoidal-based loss adaptation for task coordination
3. **Context-Aware Modules**: Task-specific feature refinement (LCEM/GCEM)

**Key Advantages**:
- 🏆 **91.82% RV Dice** | **98.54% FAZ Dice** on OCTA-500
- ✨ Superior microvascular segmentation near FAZ boundaries
- ⚡ Flexible integration with existing encoder-decoder architectures

## 🚀 Key Features
- **Multi-Scale Semantic Extraction**: Hybrid CNN-Transformer backbone captures local details and global dependencies
- **Task-Specific Context Exchange**:
  - **LCEM**: Local vascular pattern refinement
  - **GCEM**: Global FAZ boundary reasoning
- **Dynamic Weight Balancing (DWB)**:
  ```python
  α = 0.1 + 0.8*sin(epoch_progress) + 0.1*(loss_ratio)  # Dynamic task weighting


## 📊 Performance Highlights
OCTA-500 Dataset
Method	RV Dice (%)	FAZ Dice (%)
U-Net	88.23	95.69
TransUNet	90.06	97.39
Joint-Seg	90.35	98.05
S²A-Net	90.82	98.54
FAZ Perivascular Region
Method	OCTA-500	ROSE-H
VAFF-Net	97.68	79.64
S²A-Net	98.64	81.23


