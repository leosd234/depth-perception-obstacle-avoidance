# Depth Perception & Obstacle Avoidance
Real-time obstacle detection using Intel RealSense D435i depth camera and YOLOv8.
Detects objects, estimates their distance, and color-codes them by danger level.

## Demo
🚧 *Demo gif coming soon*

## How it works
1. RealSense D435i captures aligned color + depth frames at 30fps
2. YOLOv8 detects objects in the color frame
3. Depth frame measures the exact distance to each detected object
4. Bounding boxes are color-coded by distance:
   - 🔴 Red — closer than 1m (danger)
   - 🟠 Orange — 1m to 2m (caution)
   - 🟢 Green — beyond 2m (clear)

## Requirements
- Intel RealSense D435i
- Python 3.8+

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/detector.py
```

## Tech Stack
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy

## Author
Jose Leonardo Salazar — [GitHub](https://github.com/leosd234)