# autonomous-driving-vision-system
Autonomous vehicle system using deep learning for adaptive cruise control. Combines FCN-ResNet50 segmentation (90.71% accuracy) and YOLOv8 detection for real-time scene understanding. Features hierarchical decisions, weather adaptation, and traffic compliance. Python, PyTorch, OpenCV on CARLA dataset.


# Adaptive Cruise Control System with Computer Vision

An autonomous vehicle navigation system using deep learning for adaptive cruise control. Combines FCN-ResNet50 semantic segmentation and YOLOv8 object detection for real-time scene understanding.

## Overview

This project implements an intelligent cruise control system that processes road scenes in real-time to make autonomous driving decisions. The system combines two neural network models to understand the environment and adapt vehicle speed based on traffic conditions, weather, and obstacles.

**Key Features:**
- Real-time semantic segmentation (12 classes: road, vehicles, pedestrians, signs, etc.)
- Object detection for traffic signs, signals, and vehicles
- Hierarchical safety-first decision logic
- Weather-based speed adaptation (sunny, rainy, foggy, night)
- CPU-optimized for deployment without GPU

**Performance:**
- 90.71% pixel accuracy in semantic segmentation
- 48.99% mean IoU across all classes
- Real-time processing on CPU

## System Architecture

```
Input Video Frame
       ↓
       ├─→ Semantic Segmentation (FCN-ResNet50)
       │   └─→ 12 scene classes
       │
       └─→ Object Detection (YOLOv8)
           └─→ Vehicles, signs, pedestrians
       ↓
Decision System
       ↓
Speed Control Output
```

### Components

1. **Segmentation Module**: FCN-ResNet50 for pixel-level scene classification
2. **Detection Module**: YOLOv8 Nano for object detection
3. **Decision System**: Hierarchical logic with 5 priority levels

## Results

### Semantic Segmentation Performance

| Metric | Value |
|--------|-------|
| Pixel Accuracy | 90.71% |
| Mean IoU | 48.99% |
| Test Loss | 0.2616 |

**Top Classes (IoU):**
- Road: 90.92%
- Unlabeled: 92.49%
- Building: 80.15%
- Car: 78.37%

### Weather-Based Speed Adaptation

| Condition | Speed Multiplier | Max Speed |
|-----------|-----------------|-----------|
| Sunny | 100% | 50 km/h |
| Cloudy | 90% | 45 km/h |
| Night | 85% | 42 km/h |
| Rainy | 75% | 38 km/h |
| Foggy | 65% | 32 km/h |

### Decision Logic Priority

1. **Emergency Stop**: Vehicle too close (area ratio > 0.15) → 0 km/h
2. **Braking**: Vehicle approaching (0.08-0.15) → Reduce speed
3. **Traffic Signs**: Stop/red light compliance → 0 km/h
4. **Weather**: Adapt speed based on conditions
5. **Cruise**: Normal driving → 50 km/h baseline

## Getting Started

### Prerequisites

```bash
Python 3.8+
PyTorch 2.0+
torchvision
OpenCV
NumPy
Matplotlib
Pillow
ultralytics (YOLOv8)
```

### Installation

```bash
# Clone repository
git clone https://github.com/KovacsErik-Gh/adaptive-cruise-control-vision.git
cd adaptive-cruise-control-vision

# Install dependencies
pip install -r 
```

### Usage

#### Training the Segmentation Model

```python
# Open Jupyter Notebook
jupyter notebook notebooks/cruise_control.ipynb

# Run training cells
# Model will be saved to models/fcn_resnet50_carla.pth
```

#### Running the Simulator

```python
# Load trained models and run inference on test videos
# See notebook for full implementation
```

## 📁 Project Structure

```
adaptive-cruise-control-vision/
├── README.md
├── requirements.txt
├── notebooks/
│   └── cruise_control.ipynb       # Main implementation
├── models/
│   ├── fcn_resnet50_carla.pth    # Segmentation model
│   └── yolov8_carla.pt           # Detection model
├── data/
│   ├── train/                     # Training videos
│   ├── val/                       # Validation videos
│   └── test/                      # Test videos
├── results/
│   ├── segmentation_results.png
│   ├── detection_demo.png
│   └── decision_scenarios/
└── docs/
    ├── technical_report.pdf
    └── presentation.pptx
```

## 🔧 Technical Details

### Semantic Segmentation
- **Architecture**: FCN-ResNet50 (Fully Convolutional Network)
- **Input Size**: 96×192 pixels (optimized for CPU)
- **Output**: 12-class pixel-wise predictions
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam (lr=1e-4)
- **Training**: 2 epochs on CARLA dataset

### Object Detection
- **Model**: YOLOv8 Nano
- **Detects**: Cars, trucks, buses, pedestrians, bicycles, traffic lights, stop signs, speed limits
- **Confidence Threshold**: 0.5
- **NMS IoU**: 0.45

### Dataset
- **Source**: CARLA Semantic Segmentation Dataset
- **Videos**: 28 videos across diverse scenarios
- **Split**: 70% train / 15% validation / 15% test
- **Conditions**: Multiple weather types (sunny, rainy, foggy, night)

## Academic Context

Developed as a course project for **Knowledge-Based Systems** at the Technical University of Cluj-Napoca (UTCN), Faculty of Automation and Computer Engineering.

**Course**: Sisteme Bazate pe Cunoaștere  
**Semester**: Year 3, Semester 1  
**Supervisor**: Conf. Dr. Ing. Roxana Rusu Both

## Future Improvements

- [ ] Increase training resolution to 256×512 for better small object detection
- [ ] Extend training to 10-20 epochs for improved convergence
- [ ] Implement 3D depth estimation for accurate distance measurement
- [ ] Add trajectory planning for lane changes and turns
- [ ] Integrate with real CARLA simulator for dynamic testing
- [ ] Test on real-world dashcam footage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

**Erik Kovács**
- University: Technical University of Cluj-Napoca
- Faculty: Automation and Computer Engineering
- Email: kovacs.erik171004@gmail.com
- LinkedIn: www.linkedin.com/in/kovacs-erik-598691397

## Acknowledgments

- CARLA team for the autonomous driving simulator and dataset
- PyTorch and Ultralytics teams for excellent deep learning frameworks
- Course supervisor for guidance and support

---

If you found this project helpful, please consider giving it a star!
