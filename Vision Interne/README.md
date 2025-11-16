Driver Monitoring Module — SafeDrive AI

This module detects drowsiness, driver distraction, and calculates a Driver State Score (0–1) using MediaPipe FaceMesh.
It is part of the larger SafeDrive AI system for accident prevention and pre-crash prediction.

 What This Module Does
* Detects Drowsiness
Uses Eye Aspect Ratio (EAR)
Detects eye closure over multiple frames
Triggers a DROWSY alert

* Detects Distraction
Detects head turning (left/right)
Alerts if the driver looks away for more than 3 seconds

* Calculates a Final Score (0–1)
Based on:
time with eyes closed
time distracted
overall attention stability
Score is printed at the end when the program closes

How It Works
MediaPipe FaceMesh detects 468 face landmarks.
Eye landmark distances → EAR for drowsiness.
Nose-to-center alignment → head angle for distraction.
Counters track how long the driver:
- had eyes closed
- was looking away

At the end score is calculated (0–1)


How to Run
Make sure you have MediaPipe + OpenCV:
  pip install mediapipe opencv-python
Then run:
  python driver_monitor.py
Press Q to quit the monitoring window and print the final driver score.

Requirements :
Python 3.8+
OpenCV
MediaPipe
Webcam (720p recommended)
