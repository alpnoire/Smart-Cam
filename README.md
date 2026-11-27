# Smart Cam

AI-powered real-time surveillance using Python, OpenCV and YOLO26.

## Features
- Real-time object detection
- Object tracking with unique IDs
- Custom restricted-zone selection
- Zone breach detection
- Visual and audio alerts
- Automatic evidence image capture
- CSV event logging
- Live FPS and detection statistics
- Manual snapshots

## Technology Stack
- Python
- YOLO26
- Ultralytics
- OpenCV

## Installation

Clone the repository:

    git clone YOUR_REPOSITORY_URL
    cd Smart-Cam

Create a virtual environment:

    python -m venv .venv

Activate it:

    .venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

## Run

    python smart_cam2.py

## Command-Line Options

Use another camera:

    python smart_cam2.py --source 1

Process a video:

    python smart_cam2.py --source video.mp4

Use a larger model:

    python smart_cam2.py --model yolo26s.pt

Detect multiple classes:

    python smart_cam2.py --classes 0 2 16

Adjust confidence:

    python smart_cam2.py --conf 0.5

## Controls

| Input | Action |
|---|---|
| Mouse drag | Draw restricted zone |
| C | Clear zone |
| R | Reset counts and alert cooldowns |
| S | Save snapshot |
| Q | Quit |

## Alert Workflow

When an object enters the restricted zone:
1. A visual warning appears.
2. An evidence image is saved.
3. The event is recorded in alerts/log.csv.
4. A Windows beep alert is played.

A 10-second cooldown is applied per tracking ID.

## Future Improvements
- Telegram notifications
- Web monitoring dashboard
- Remote surveillance
- Historical analytics

## Author

Developed as an AI and Computer Vision project.

**Smart Cam: Making everyday cameras smarter.**
