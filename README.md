# Face Recognition System (LBPH + FaceNet + Ensemble)

A modular face recognition system supporting classical (LBPH), modern embedding-based (FaceNet), and ensemble inference modes with proper train/validation/test evaluation and live webcam deployment.

## Features
- Person-wise train/validation/test split
- LBPH baseline face recognition model
- FaceNet embedding-based recognition using cosine similarity
- Open-set recognition with unknown identity rejection
- Ensemble strategy combining FaceNet and LBPH
- Real-time live webcam inference with selectable modes

## Project Structure
Face_Recognition/  
├── src/                 # Core logic  
├── data/                # Dataset (ignored in Git for privacy)  
├── main.py              # Entry point  
├── config.yaml  
├── requirements.txt  
└── README.md  

## Installation
pip install -r requirements.txt

## Usage 
## Offline Evaluation
python main.py lbph
python main.py facenet
python main.py compare

## Live Webcam Inference
python main.py live_lbph
python main.py live_facenet
python main.py live_ensemble

Press q to exit any live mode.

## Models
    LBPH: Classical texture-based face recognition baseline.

    FaceNet: Pretrained deep face embeddings with cosine similarity matching.

    Ensemble: FaceNet-first decision strategy with LBPH as a secondary consistency check.

## Notes
Face images are not included in this repository for privacy reasons.

Thresholds are selected empirically using validation data.

Designed for small datasets with limited samples per identity.