from ultralytics import YOLO
import cv2
import os
import time
import requests

# Configuration
topic = "flowers"  # Replace 'flowers' with the desired topic name
model_path = "yolo11n.pt"  # Path to YOLO model
webcam_index = 0  # Default webcam index
history_dir = "history"  # Subdirectory for saving frames
test_interval = 3  # Time (in seconds) between detection attempts
upload_timeout = 10  # Time (in seconds) before the next upload is allowed
preview_enabled = False # Enable or disable preview frames

print("Loading model, this may take a moment...")

# Load the YOLO model
model = YOLO(model_path)  # Ensure the model file is present in the same directory or specify its path

# Open the Mac webcam
webcam = cv2.VideoCapture(webcam_index)  # 0 is the default camera index

if not webcam.isOpened():
    print("Error: Could not open webcam.\nTry changing the camera index.")
    exit()

# Create the 'history' subdirectory if it doesn't exist
os.makedirs(history_dir, exist_ok=True)

print("Running... Press 'Ctrl+C' to stop.")

last_uploaded_time = 0  # Time of the last successful upload
last_test_time = 0  # Time of the last detection attempt

try:
    while True:
        current_time = time.time()

        # Skip detection during timeout period
        if current_time - last_uploaded_time < upload_timeout:
            time.sleep(0.1)  # Sleep briefly to avoid busy-waiting
            continue

        # Perform detection only at the specified interval
        if current_time - last_test_time >= test_interval:
            # Read a frame from the webcam
            ret, frame = webcam.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Perform object detection
            results = model(frame)  # Pass the frame to the YOLO model
            annotated_frame = results[0].plot()  # Draw the detections on the frame

            
            # Add timestamp and battery info to the frame
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Current date and time
            battery_status = get_battery_status()  # Get battery percentage

            # Add text overlays
            cv2.putText(annotated_frame, timestamp, (10, 20), font, font_scale, font_color, thickness)
            cv2.putText(annotated_frame, battery_status, (10, 40), font, font_scale, font_color, thickness)

            # Check if a person is detected
            person_detected = any(box.cls == 0 for box in results[0].boxes)  # Check for 'person' detections

            # If a person is detected, save and upload the frame
            if person_detected:
                timestamp = time.strftime("%Y%m%d_%H%M%S")  # Create a timestamp for the filename
                output_path = os.path.join(history_dir, f"frame_{timestamp}.jpg")
                cv2.imwrite(output_path, annotated_frame)  # Save the annotated frame

                # Upload the frame to ntfy.sh
                try:
                    with open(output_path, 'rb') as file:
                        response = requests.put(
                            f"https://ntfy.sh/{topic}",
                            data=file,
                            headers={"Filename": f"frame_{timestamp}.jpg"}
                        )
                        if response.status_code == 200:
                            print(f"Frame uploaded to ntfy.sh topic '{topic}'.")
                        else:
                            print(f"Failed to upload frame. Status code: {response.status_code}")
                except Exception as e:
                    print(f"Error during upload: {e}")

                # Update the last uploaded time and start the timeout
                last_uploaded_time = current_time

            if preview_enabled:
                cv2.imshow("YOLO Person Detection", annotated_frame)

            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Update the last test time
            last_test_time = current_time

except KeyboardInterrupt:
    print("\nStopping...")


# Release resources
webcam.release()
cv2.destroyAllWindows()
