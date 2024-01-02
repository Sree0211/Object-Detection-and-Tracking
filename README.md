# Object Tracking with Qt GUI

## Overview

This project implements real-time object tracking of a highway scene using OpenCV and displays the results in a Qt GUI. The main features include:

- Object detection using a background subtractor.
- Displaying the original video feed alongside the masked video with detected objects.
- Pause and resume functionality for video playback.
- Progress bar indicating the current frame's position in the video.

## Prerequisites

Before running the project, make sure you have the following dependencies installed:

- Python3
- PyQt5
- OpenCV

Install the required Python packages using:

```bash
pip install PyQt5 opencv-python
```

## How it Works
ObjectTracking_Counting.py: This file contains the implementation of the object tracking algorithm using OpenCV. It detects objects, applies a mask, and provides the frames for display.

ObjectTracking_Window.py: The main application file that initializes the Qt GUI, sets up video panels, handles video playback, and connects with the object tracking implementation.

## Sample Output

https://github.com/Sree0211/Object-Detection-and-Tracking/assets/43269126/89ab12cf-3110-411a-8ff5-2cde69d97d24

