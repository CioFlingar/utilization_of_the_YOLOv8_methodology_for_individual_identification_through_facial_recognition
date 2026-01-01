# Person Identification with Facial Recognition using YOLOv8

This repository contains the implementation of a real-time **Person Identification System** using facial recognition, based on the **YOLOv8** object detection model. The system is designed to detect and identify individuals in real-time using facial features, tailored for robust performance in diverse environments.

## Features

-   Real-time face detection and person identification
-   High-speed inference with YOLOv8
-   Accurate and secure identification
-   Pre-processing with facial alignment and augmentation
-   Custom-trained model on annotated facial datasets
-   Comprehensive evaluation with precision, recall, F1-score, and mAP
-   Experimental results and case studies included

## Model Architecture

-   **YOLOv8** for efficient object detection
-   Custom CNN-based facial feature extractor
-   Trained on labeled datasets with person-specific annotations
-   Softmax classification for identity prediction

## Dataset

-   **Source**: Own Custom Dataset
-   Contains images with varied poses, lighting conditions, and occlusions, teared, noise, and blurr
-   Data split: `Train (70%)`, `Validation (15%)`, `Test (15%)`
-   Pre-processing: Face alignment, resizing, augmentation (flip, rotate, crop)

## Usage

Run real-time identification (webcam):

```bash
python identify.py --source 0 --weights runs/train/exp/weights/best.pt

```

Run on an image:

```bash
python identify.py --source path/to/image.jpg --weights runs/train/exp/weights/best.pt

```

Train a custom model:

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640

```

## 📊 Evaluation

Metrics used for evaluation:

-   [mAP@0.5](mailto:mAP@0.5)
-   Precision, Recall, F1-score
-   Confusion Matrix

Run evaluation:

```bash
python evaluate.py --weights runs/train/exp/weights/best.pt --data data.yaml

```

## 📝 Thesis Summary

This project, part of the thesis **"Person Identification with Facial Recognition on Deep Learning"**, investigates real-time person identification using YOLOv8. The model integrates face detection with a classification module trained on facial embeddings, achieving high accuracy and robustness in challenging scenarios.

## 📚 Technologies Used

-   Python
-   YOLOv8 (Ultralytics)
-   OpenCV
-   PyTorch
-   NumPy, Matplotlib, Scikit-learn

## 📄 License

© 2026 Wahid Hasan. All rights reserved.

## 👨‍💻 Authors

1.  **Walid Hasan**
2.  Sumaya Mahira 
3.  Golam Shakib Hossen 
4.  Harun-AR-Rashid
5.  Torikuzzaman Mollah

## Citation
If you use this code or methods, please cite our IEEE conference paper:

```bash
W. Hasan, S. Mahira, G. Shakib Hosen, M. Harun-Ar-Rashid, and T. Mollah, 
“Utilization of the YOLOv8 methodology for individual identification through facial recognition,” 
in *Proc. 2024 IEEE 13th Int’l Conf. Electrical and Computer Engineering (ICECE)*, Dec. 2024, pp. 651–656. 
doi:10.1109/ICECE64886.2024.11024852.
```
##### Link to IEEE Xplore: 
```bash
DOI:10.1109/ICECE64886.2024.11024852
 ```

[IEEE Xplore Direct Link](https://doi.org/10.1109/ICECE64886.2024.11024852)


## 📌 Acknowledgements
- Ultralytics YOLOv8 for the base object detection framework
- IEEE ICECE 2024 for publishing the original research

