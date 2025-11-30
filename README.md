# 🐮 AI Cow Counting & Tracking System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-red)
![Status](https://img.shields.io/badge/Status-POC-orange)

> A real-time Computer Vision application that detects, tracks, and counts livestock using YOLOv8 and OpenCV.

## 📖 Introduction

As a **Backend Developer** typically working with Django and Database Architectures, I wanted to explore the challenges of unstructured data processing (Video/Images). 

This project is a Proof of Concept (POC) to build a monitoring system for agriculture. It addresses specific challenges such as distinguishing animals from static objects (rocks, bushes) and maintaining accurate counts in real-time.

## 🚀 Key Features

- **Object Detection:** Utilizes `YOLOv8n` (Nano) for fast inference on CPU.
- **Real-time Tracking:** Implements tracking algorithms to assign unique IDs to each animal.
- **False Positive Filtering:** Optimized Confidence Threshold logic to prevent misidentifying static objects (e.g., rocks) as cows.
- **Real-time Counting:** Displays the actual number of animals in the frame dynamically.
- **Video Export:** processing and saving results to MP4 format.

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Core AI Model:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Computer Vision:** OpenCV (`cv2`)
- **Tracking Support:** `lapx` (Linear Assignment Problem solver for Windows)

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/quocbao2004/cow-counting-ai.git
   cd cow-counting-ai
   ```
   Run code
   ```bash
   python main.py --source cows.mp4 --show --save output.mp4 
