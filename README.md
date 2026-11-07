# CodeAlpha Object Detection and Tracking

A real-time object detection and tracking application developed for the CodeAlpha Data Science Internship. This project uses the YOLOv8 medium model and ByteTrack to accurately identify, track, and count unique objects across video files and live webcam feeds.

## Core Features

* Dual Input Modes: Seamlessly switch between processing saved video files and a live webcam feed via a terminal prompt.
* Robust Object Tracking: Utilizes ByteTrack with a custom configuration to retain object IDs during temporary occlusions (up to 90 frames), eliminating ID flickering.
* Targeted Detection: Filtered to specifically track people, cars, motorcycles, buses, and trucks while ignoring background clutter.
* Analytics Overlay: Displays real-time system processing speed (FPS) and a running count of unique objects detected in the session.
* Automated Video Export: Automatically saves the processed output as a newly encoded MP4 file for easy sharing and review.

## Prerequisites

* Python 3.8 or higher
* A virtual environment (recommended)

## Installation

1. Clone or download this repository to your local machine.
2. Open a terminal in the project directory and create a virtual environment:
   python -m venv venv
3. Activate the virtual environment:
   Windows: venv\Scripts\activate
   Mac/Linux: source venv/bin/activate
4. Install the required dependencies:
   pip install -r requirements.txt

## Usage

1. Place any test videos (e.g., test_video.mp4) into the root project directory.
2. Run the main tracking script:
   python object_tracker.py
3. Follow the terminal prompt:
   * To process a video file, type the exact filename and press Enter.
   * To run live recognition (webcam), simply press Enter without typing anything.
4. Press 'q' at any time while the video window is active to stop processing early.

## Output

The script automatically saves the final processed video in the same directory.
* If a file was processed, the output is prefixed with "Tracked_" (e.g., Tracked_test_video.mp4).
* If the webcam was used, the output is saved as "CodeAlpha_Live_Output.mp4".