# Security Camera with YOLOv11 Detection and Overlay

This Python script implements a security camera using the YOLOv11 object detection model. It detects people using the device's webcam, overlays battery status, date, and time onto the frames, and uploads detected frames to [ntfy.sh](https://ntfy.sh) for remote notification. Annotated frames are also saved locally in a `history` folder for record-keeping.

This entire repository was written using ChatGPT from OpenAI.

[GITHUB](https://github.com/Delamcode/Yolo-Cam)

---

## **Important: Change the `topic` for ntfy.sh Uploads**

At the top of the script, there is a variable called `topic`. This variable specifies the ntfy.sh topic where the script will upload frames that contain detected people. To avoid mixing notifications with others, **be sure to change this `topic` to a unique name** that you choose. Look at the [ntfy.sh documentation](https://docs.ntfy.sh/#step-1-get-the-app) for more information about this topic.

Example:
```python
topic = "my-unique-topic"  # Replace 'my-unique-topic' with your own desired topic
```

---

## Features

- **Person Detection**: Detects people in webcam frames using the YOLOv11 model.
- **Real-Time Annotations**: Draws bounding boxes, labels, battery percentage, current time, and date on the detected frames.
- **Efficient Execution**:
  - Skips processing during a cooldown period (`upload_timeout`) after an upload.
  - Detects frames at a configurable interval (`test_interval`) when not in timeout.
- **Frame Uploads**: Automatically uploads annotated frames containing detected persons to a configurable [ntfy.sh](https://ntfy.sh) topic.
- **Local Archiving**: Saves all uploaded frames in a `history` directory.
- **Graceful Shutdown**: Cleanly exits on `Ctrl+C`.
- **Optional Preview**: Optionally shows a running window.
- **Priority Notifications**: Changes priority based on detection confidance.

---

## Requirements

- **Python**: 3.8 or later
- **Packages**:
  - `ultralytics` (for YOLOv11 model)
  - `opencv-python` (for video processing)
  - `psutil` (for battery status)
  - `requests` (for HTTP requests)

Install the required packages:
```bash
pip install ultralytics opencv-python psutil requests
```
It is also recomended to use a virtual enviornment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install ultralytics opencv-python psutil requests
```

---

## Configuration

You can customize the script using the following variables:

- `topic`: The [ntfy.sh](https://ntfy.sh) topic for uploading detected frames. Replace `flowers` with your desired topic.
- `model_path`: Path to the YOLOv11 model file. The model will be downloaded on first run of the program.
- `history_dir`: Directory where detected frames will be saved locally.
- `test_interval`: Time (in seconds) between detection attempts when not in a timeout. Default: `10 seconds`.
- `upload_timeout`: Time (in seconds) during which the script skips processing frames after an upload. Default: `3 seconds`.
- `preview_enabled`: Boolean to show or hide the preview. Default: `True`.
- `highpriority_confidance_requirement`: Confidance requirment for a high priority notification. Default: `0.45`.

---

## How It Works

1. **Object Detection**:
   - Captures frames from the webcam.
   - Processes them using the YOLOv11 model.
   - Checks for `person` detections in the frame.

2. **Frame Annotation**:
   - Draws bounding boxes around detected objects.
   - Overlays the current battery level, date, and time at the top-left corner.

3. **Frame Handling**:
   - If a person is detected:
     - Saves the annotated frame in the `history` directory.
     - Uploads the frame to the specified `ntfy.sh` topic.

4. **Power Efficiency**:
   - Skips frame processing during the `upload_timeout` period to save resources.

---

## How to Run

1. Clone the repository or download the script.
2. Run the script:
   ```bash
   python main.py
   ```
   Note that this will take some time on first start, as it is downloading the model.
3. Press `Ctrl+C` to stop the script.

---

## Example Output

1. **On Frame Detection**:
   - The frame will be saved locally in the `history` folder.
   - The frame will include:
     - Bounding boxes and labels for detected objects.
     - Timestamp (e.g., `2024-11-23 14:15:00`).
     - Battery status (e.g., `Battery: 85%`).

2. **Frame Upload**:
   - Frames with detected persons will be uploaded to the configured [ntfy.sh](https://ntfy.sh) topic.
   - Example upload log:
     ```
     Frame uploaded to ntfy.sh topic 'flowers'.
     ```

---

## Directory Structure

```plaintext
project/
│
├── main.py             # Main script
├── yolo11n.pt          # YOLOv11 model file (download on first run)
└── history/            # Folder to store detected frames (auto-created)
```

---

## Dependencies Explained

1. **ultralytics**:
   - For downloading, loading, and using the YOLOv11 object detection model.

2. **opencv-python**:
   - For webcam access and frame processing.

3. **psutil**:
   - To retrieve the device's battery percentage.

4. **requests**:
   - For uploading frames to the `ntfy.sh` service.

---

## Notes

- The YOLOv11 model (`yolo11n.pt`) should be pre-trained and support the `person` class (`cls == 0`).
- Ensure that your device supports battery monitoring if you want the battery overlay (laptops typically do).
- [ntfy.sh](https://ntfy.sh) is a free notification service that supports file uploads. Check its documentation for more details.

---

## Customization

- **Change Upload Topic**:
  - Modify the `topic` variable to change the ntfy.sh topic for uploading frames.

- **Detection Classes**:
  - Update the detection logic to handle other classes by changing `box.cls == 0` to the desired class ID.

- **Overlay Appearance**:
  - Adjust the `font`, `font_scale`, `font_color`, and `thickness` variables to customize the overlay text style.

---

## Troubleshooting

1. **Webcam Not Found**:
   - Ensure your webcam is connected and accessible.
   - Update the `webcam_index` variable if the default index (0) doesn’t work.

2. **Battery Status Unavailable**:
   - On devices without battery sensors, the script will display `Battery: N/A`.

3. **Upload Issues**:
   - Ensure your device has internet access.
   - Check the [ntfy.sh](https://ntfy.sh) service status.

4. **Submit an Issue**
   - Feel free to submit an issue [here](https://github.com/Delamcode/Yolo-Cam/issues/new/)

---

## License

Currently N/A

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLOv11 model.
- [OpenCV](https://opencv.org/) for video and image processing.
- [ntfy.sh](https://ntfy.sh) for free notifications and file upload services.
