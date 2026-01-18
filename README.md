#  Eye Blink & Drowsiness Detection Project

This project implements an **Eye Blink and Drowsiness Detection system** using
**machine learning and computer vision techniques**.  
The system monitors eye activity through a webcam and detects **eye blinks**
and **prolonged eye closure** to identify signs of fatigue.  
When drowsiness is detected, a **sound alert** is triggered in real time.

---

##  Project Objective

The objective of this project is to build a **real-time fatigue detection system**
that can monitor eye behavior and alert the user when drowsiness is detected,
helping to improve safety and awareness.

---

##  ML / CV Approach

- Facial landmarks are detected using **MediaPipe Face Mesh**
- Eye landmarks are extracted from the detected face
- **Eye Aspect Ratio (EAR)** is calculated to determine eye state
- Eye blinks are counted using EAR thresholding
- Drowsiness is detected when eyes remain closed beyond a time threshold

---

##  Technologies Used

Python  
OpenCV  
MediaPipe  
NumPy  
Playsound  

---

##  How the System Works

- Webcam captures live video frames  
- Face and eye landmarks are detected in real time  
- EAR value is calculated for both eyes  
- If EAR falls below a threshold, eyes are considered closed  
- Prolonged eye closure is classified as drowsiness  
- A sound alert is triggered to warn the user  

---

##  Features

- Real-time webcam-based eye monitoring  
- Eye blink detection and counting  
- Drowsiness detection using EAR  
- Face and eye bounding boxes  
- Sound alert on fatigue  
- Lightweight and efficient implementation  

---

##  Project Structure

eye-drowsiness-detection/
├── drowsiness.py
├── assets/
│   └── alert.wav
├── requirements.txt
└── eye_drowsiness_env/

Author

Aditya Singh Bhadauria

Usage Note

This project is developed strictly for academic and educational purposes.
