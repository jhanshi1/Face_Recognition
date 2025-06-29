# 🎯 Face Recognition System using OpenCV

This is a beginner-friendly face recognition system built using Python and OpenCV. The project demonstrates how to train a computer to recognize multiple faces from image data and make predictions in real-time through webcam.

---
 Features

- **Face Detection** using Haar Cascade Classifier
- **Real-time Recognition** through webcam feed
- **Custom Image Dataset Support**
- **Label Training & Model Saving**
- Works on both grayscale and RGB images

---

## 📁 Project Structure

```bash
face_recognition_system/
│
├── images_face/             # 📂 Training images categorized in folders (1 folder per person)
├── face_trained.yml         # 🧠 Trained face recognizer model
├── face_recognition.py      # 🖥️ Real-time face recognition script
├── faces_train.py           # 🏋️‍♂️ Training script for recognizer
├── harr_cas.xml             # 🔍 Haar Cascade XML file for face detection
└── README.md                # 📄 Project overview and instructions
