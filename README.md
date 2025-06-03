# YOLO Model Quantization Performance Comparison Tool

A comprehensive tool for quantizing YOLO11 models and comparing their performance against original models. This tool provides detailed metrics on model size reduction, inference speed improvements, and accuracy preservation through quantization.

## Features

- **Model Quantization**: Dynamic quantization of YOLO11 models
- **Performance Benchmarking**: Detailed timing comparisons between original and quantized models
- **Accuracy Analysis**: Model-to-model accuracy comparison with IoU calculations
- **Ground Truth Evaluation**: Support for XML annotation files with precision/recall/F1-score metrics
- **Memory Footprint Analysis**: Comprehensive memory usage monitoring
- **Production Mode**: Optimized settings for production environments

## Requirements

### Python Version
- **Python 3.8 - 3.11** (Recommended: Python 3.10 or 3.11)
- **Note**: Python 3.11 should work perfectly with this code

### System Requirements
- **CPU**: Multi-core processor (threading optimizations applied automatically)
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: At least 1GB free space for models and results

## Installation

### Install Required Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Computer vision and YOLO
pip install ultralytics

# System monitoring and utilities
pip install psutil

# Image processing
pip install Pillow

# Data handling
pip install numpy

# All dependencies in one command:
pip install torch torchvision torchaudio ultralytics psutil Pillow numpy
```

### Alternative: Using requirements.txt
Create a `requirements.txt` file with:
```txt
torch>=1.13.0
torchvision>=0.14.0
torchaudio>=0.13.0
ultralytics>=8.0.0
psutil>=5.9.0
Pillow>=9.0.0
numpy>=1.21.0
```

Then install:
```bash
pip install -r requirements.txt
```

## Configuration

### Directory Structure
```
your-project/
├── paste.txt (your main script)
├── HoloSelecta/ (optional - your dataset directory)
│   ├── image1.jpg
│   ├── image1.xml
│   ├── image2.jpg
│   └── image2.xml
├── test6_prod_results_100/ (auto-created for results)
├── yolo11s.pt (downloaded automatically)
└── yolo11s_quantized.pt (created during quantization)
```

### Configuration Parameters
At the top of the script, you can modify these constants:

```python
NUM_TEST_IMAGES = 10          # Number of test images to process
TEST_IMAGES_DIR = "HoloSelecta"  # Directory containing your images
RESULTS_DIR = "test6_prod_results_100"  # Results output directory
BATCH_SIZE = 10               # Processing batch size
MODEL_INPUT_SIZE = 640        # Model input resolution
CONFIDENCE_THRESHOLD = 0.25   # Detection confidence threshold
IOU_THRESHOLD = 0.45          # IoU threshold for NMS
MODEL_PATH = "yolo11s.pt"     # YOLO model file
PRODUCTION_MODE = True        # Set to False for detailed benchmarking
```

## Usage

### With Custom Dataset
1. Create your dataset directory (e.g., `HoloSelecta/`)
2. Add your images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`)
3. Add corresponding XML annotations (optional, same filename as images)
4. Update `TEST_IMAGES_DIR` in the script if using a different directory name
5. Run the script

### Production vs Development Mode

**Production Mode** (`PRODUCTION_MODE = True`):
- Single inference run per image
- Minimal calibration (3 runs)
- Optimized for speed and efficiency
- Suitable for deployment testing

**Development Mode** (`PRODUCTION_MODE = False`):
- Multiple inference runs (5) with averaging
- Extended calibration (10 runs)
- Detailed timing statistics (min/max/average)
- Better for thorough analysis

## Dataset Format

### Image Files
Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

### XML Annotations (Optional)
The tool supports Pascal VOC format XML annotations:

```xml
<annotation>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <object>
        <name>person</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>50</ymin>
            <xmax>200</xmax>
            <ymax>150</ymax>
        </bndbox>
    </object>
</annotation>
```


## Troubleshooting

### Common Issues

**1. Model Download Issues**
```bash
# If YOLO11s model fails to download, try manually:
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"
```

**2. Memory Issues**
- Reduce `NUM_TEST_IMAGES`
- Close other applications
- Use smaller `MODEL_INPUT_SIZE` (e.g., 320 instead of 640)

**3. Import Errors**
```bash
# Reinstall ultralytics
pip uninstall ultralytics
pip install ultralytics

# For torch issues on specific systems:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

**4. Permission Errors**
- Ensure you have write permissions in the working directory
- Run with appropriate permissions or change output directories

### Python 3.11 Compatibility
This code is fully compatible with Python 3.11. If you encounter any issues:

1. Ensure all dependencies are up to date:
```bash
pip install --upgrade torch torchvision ultralytics psutil Pillow numpy
```

2. Check PyTorch compatibility:
```bash
python -c "import torch; print(torch.__version__)"
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are correctly installed
3. Ensure Python 3.8-3.11 is being used
4. Check that you have sufficient system resources

---

**Note**: This tool is optimized for CPU inference. For GPU acceleration, modify the device settings throughout the code from `'cpu'` to `'cuda'` where appropriate.
