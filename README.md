# Computer Vision Final Project

## Team Members
1. Member 1 — Tasks 1 & 4 (Selective Enhancement + Document Cleaning)
2. Member 2 — Tasks 2 & 3 (X-Ray Sharpening + Auto Enhancement)
3. Member 3 — Task 5 (Panorama Stitching)
4. Member 4 — Tasks 6 & 7 (Object Recognition + Depth Approximation)
5. Member 5 — Task 8 + Report (HDR Imaging)

## Project Structure

```
├── cv_utils/           # Shared utility functions
│   └── helpers.py      # I/O, display, analysis, metrics
├── tasks/              # One script per task
│   ├── task1_selective_enhancement.py
│   ├── task2_xray_sharpening.py
│   ├── task3_auto_enhancement.py
│   ├── task4_document_cleaning.py
│   ├── task5_panorama_stitching.py
│   ├── task6_object_recognition.py
│   ├── task7_depth_approximation.py
│   └── task8_hdr_imaging.py
├── images/             # Input images (per task)
├── outputs/            # Output images (per task)
└── report/             # Final report
```

## Setup

```bash
pip install opencv-python numpy matplotlib
```

## Running Tasks

```bash
# Run any task from the project root:
python tasks/task1_selective_enhancement.py
python tasks/task2_xray_sharpening.py
# ... etc.
```

## Dependencies
- Python 3.10+
- OpenCV (`opencv-python` or `opencv-contrib-python`)
- NumPy
- Matplotlib
