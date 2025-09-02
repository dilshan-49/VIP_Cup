# VIP Cup - Faster R-CNN Object Detection

A comprehensive implementation of Faster R-CNN models for object detection, specifically designed for detecting birds and drones. This project compares the performance of different backbone architectures (ResNet-18 vs ResNet-50) and provides detailed evaluation metrics.

## 🎯 Project Overview

This repository contains implementations of Faster R-CNN object detection models using PyTorch. The project focuses on:

- **Bird and Drone Detection**: Specialized object detection for aerial objects
- **Architecture Comparison**: Performance analysis between ResNet-18 and ResNet-50 backbones
- **Model Evaluation**: Comprehensive evaluation with metrics and visualizations
- **Custom Implementation**: Custom ResNet-18 backbone integration with Faster R-CNN

## 📁 Repository Structure

```
VIP_Cup/
├── fasterRCNN.ipynb           # Main training notebook
├── model_evaluation.ipynb     # Model evaluation and comparison
├── results/                   # Model performance results
│   ├── resnet18_results.json  # ResNet-18 detection results
│   └── resnet50_results.json  # ResNet-50 detection results
├── .gitignore                 # Git ignore file
├── .vscode/                   # VS Code settings
├── VIP_Cup.code-workspace     # VS Code workspace configuration
└── README.md                  # This file
```

## 🚀 Features

- **Faster R-CNN Implementation**: Complete implementation with custom backbone support
- **Dual Architecture Support**: 
  - ResNet-18 backbone (lightweight, faster inference)
  - ResNet-50 backbone (higher accuracy, more parameters)
- **Automated Dataset Handling**: Downloads and processes dataset from Google Drive
- **YOLO Format Support**: Handles YOLO format annotations with conversion to Faster R-CNN format
- **Comprehensive Evaluation**: 
  - Performance metrics comparison
  - Visualization of predictions vs ground truth
  - Speed vs accuracy trade-off analysis
- **Result Persistence**: Saves detection results in JSON format for further analysis

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended)
- Sufficient storage space for dataset (~1GB)

### Required Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy matplotlib
pip install Pillow
pip install gdown  # for Google Drive downloads
pip install tqdm   # for progress bars
```

Or install via conda:

```bash
conda install pytorch torchvision -c pytorch
conda install opencv numpy matplotlib pillow tqdm
pip install gdown
```

## 📚 Usage

### 1. Training and Dataset Setup

Open and run `fasterRCNN.ipynb`:

```python
# The notebook will automatically:
# 1. Download the dataset from Google Drive
# 2. Set up the data loaders
# 3. Configure the models
# 4. Train both ResNet-18 and ResNet-50 variants
```

**Key Features of Training Notebook:**
- Automatic dataset download and extraction
- Custom ResNet-18 backbone implementation
- Data augmentation and preprocessing
- Training loop with loss monitoring
- Model checkpointing

### 2. Model Evaluation

Open and run `model_evaluation.ipynb`:

```python
# The notebook provides:
# 1. Model loading and comparison
# 2. Performance metrics calculation
# 3. Visualization of detection results
# 4. Speed vs accuracy analysis
```

**Evaluation Features:**
- Side-by-side model comparison
- Detection confidence visualization
- Performance metrics (mAP, precision, recall)
- Inference time analysis

### 3. Results Analysis

The `results/` directory contains:
- `resnet18_results.json`: Detection results from ResNet-18 model
- `resnet50_results.json`: Detection results from ResNet-50 model

Each result file includes:
- Image ID and filename
- Detected object category
- Bounding box coordinates
- Confidence scores

## 🎯 Model Architecture

### ResNet-18 Backbone
- **Advantages**: Faster inference, lower memory usage
- **Use Case**: Real-time applications, resource-constrained environments
- **Parameters**: ~11M parameters (backbone only)

### ResNet-50 Backbone
- **Advantages**: Higher accuracy, better feature extraction
- **Use Case**: High-accuracy requirements, offline processing
- **Parameters**: ~23M parameters (backbone only)

### Faster R-CNN Components
1. **Feature Extractor**: ResNet backbone with FPN
2. **RPN**: Region Proposal Network for object proposals
3. **ROI Head**: Classification and bounding box regression
4. **Anchor Generator**: Multi-scale anchor generation

## 📊 Performance Metrics

The models are evaluated on:
- **Detection Categories**: 
  - Class 1: Birds
  - Class 2: Drones
- **Metrics**:
  - Average Precision (AP)
  - Inference Speed (FPS)
  - Memory Usage
  - Model Size

### Sample Results

Based on the result files:
- **ResNet-18**: ~5,700+ detections with confidence scores
- **ResNet-50**: ~5,700+ detections with confidence scores
- **Coverage**: Diverse scenarios including various lighting and backgrounds

## 🔧 Configuration

### Dataset Configuration
- **Input Size**: 320x256 pixels
- **Format**: RGB images with YOLO format annotations
- **Classes**: Background (0), Bird (1), Drone (2)

### Training Configuration
- **Optimizer**: SGD with momentum
- **Learning Rate**: Adaptive scheduling
- **Batch Size**: Configurable based on GPU memory
- **Data Augmentation**: Standard augmentations for object detection

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dilshan-49/VIP_Cup.git
   cd VIP_Cup
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision opencv-python numpy matplotlib Pillow gdown tqdm
   ```

3. **Run training**:
   ```bash
   jupyter notebook fasterRCNN.ipynb
   ```

4. **Evaluate models**:
   ```bash
   jupyter notebook model_evaluation.ipynb
   ```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- torchvision for pre-trained models and utilities
- The research community for Faster R-CNN architecture
- Dataset contributors for providing annotated data

## 📬 Contact

For questions, suggestions, or collaborations, please open an issue in this repository.

---

**Note**: This project is designed for educational and research purposes. Please ensure you have appropriate permissions for any datasets used and comply with relevant licensing terms.