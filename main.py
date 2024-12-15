from ultralytics import YOLO
import cv2
import os
import time
import requests
import psutil
import keyboard
import json

# Configuration
topic = "flowers"  # Replace 'flowers' with the desired topic name
base_url = "https://ntfy.sh"
model_path = "yolo11n.pt"  # Path to YOLO model
webcam_index = 0  # Default webcam index
history_dir = "history"  # Subdirectory for saving frames
test_interval = 10  # Time (in seconds) between detection attempts
upload_timeout = 3  # Time (in seconds) before the next upload is allowed
preview_enabled = True # Enable or disable preview frames
highpriority_confidance_requirement = 0.45

font = cv2.FONT_HERSHEY_SIMPLEX  # Font for overlay text
font_scale = 0.5  # Font size
font_color = (0, 255, 0)  # Green color for text
thickness = 1  # Thickness of the text

used_message_ids = {}

if topic == "flowers":
    print("Remember to change the NTFY topic in the script before running! For more information, read the README")
    raise Exception("NTFY topic must be changed from default.") 

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

def get_battery_status():
    """Retrieve the battery percentage using psutil."""
    battery = psutil.sensors_battery()
    if battery:
        return f"Battery: {battery.percent}%"
    return "Battery: N/A"

def send_notification(title, timestamp, priority, topic, output_path, base_url):
    # Upload the frame to ntfy.sh
    try:
        with open(output_path, 'rb') as file:
            response = requests.put(
                f"{base_url}/{topic}",
                data=file,
                headers={"Title": title, "Tags": "rotating_light", "Filename": f"frame_{timestamp}.jpg", "Priority": f"{priority}"}
            )
            if response.status_code == 200:
                print(f"Frame uploaded to ntfy.sh topic '{topic}'.")
            else:
                print(f"Failed to upload frame. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error during upload: {e}")

def check_latest_message(topic, base_url):
    url = f"{base_url}/{topic}/json?poll=1"
    global used_message_ids

    try:
        # Fetch the data from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Handle multiple JSON objects in the response (newline-delimited JSON)
        data = []
        for line in response.text.splitlines():
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line}. Error: {e}")

        # Remove expired message IDs from used_message_ids
        current_time = time.time()
        used_message_ids = {
            msg_id: ts for msg_id, ts in used_message_ids.items() if current_time - ts <= 60
        }

        # Find the newest non-used "SEND" message
        for entry in reversed(data):  # Iterate from the newest to the oldest
            message_id = entry.get('id')
            message_content = entry.get('message')
            timestamp = entry.get('time', 0)  # Default to 0 if 'time' is missing

            # Check if the message is less than a minute old, hasn't been used, and has content "SEND"
            if (
                current_time - timestamp <= 60
                and message_id not in used_message_ids
                and message_content == "SEND"
            ):
                # Mark the message ID as used
                used_message_ids[message_id] = current_time
                return True
        
        return False  # No valid "SEND" message found within the last minute
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred when fetching the data: {e}")
        return False
    except Exception as e:
        print(f"An error occurred when processing the messages: {e}")
        return False

try:
    while True:
        try:
            current_time = time.time()
    
            # Skip detection during timeout period
            if current_time - last_uploaded_time < upload_timeout:
                time.sleep(0.1)  # Sleep briefly to avoid busy-waiting
                continue
    
            # Perform detection only at the specified interval
            if current_time - last_test_time >= test_interval:

                force_send = check_latest_message(topic, base_url)

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
    
                power_plugged = psutil.sensors_battery().power_plugged
                power_plugged_name = "Charging" if power_plugged else "Battery"

                # Add text overlays
                cv2.putText(annotated_frame, timestamp, (10, 20), font, font_scale, font_color, thickness)
                cv2.putText(annotated_frame, battery_status, (10, 40), font, font_scale, font_color, thickness)
                cv2.putText(annotated_frame, power_plugged_name, (10, 60), font, font_scale, font_color, thickness)

                # Check if a person is detected
                person_detected = any(box.cls == 0 for box in results[0].boxes)  # Check for 'person' detections
                highpriority_person_detected = any((box.cls == 0 and box.conf > highpriority_confidance_requirement) for box in results[0].boxes) 
                priority_detection = "max" if highpriority_person_detected else "low"
                title_detection = "Person Detected" if highpriority_person_detected else "Person Possibly Detected"
    
                # If a person is detected, save and upload the frame
                if person_detected or force_send:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Create a timestamp for the filename
                    output_path = os.path.join(history_dir, f"frame_{timestamp}.jpg")
                    cv2.imwrite(output_path, annotated_frame)  # Save the annotated frame

                    send_notification(title_detection, timestamp, priority_detection, topic, output_path, base_url)
    
                    # Update the last uploaded time and start the timeout
                    last_uploaded_time = current_time
    
                if preview_enabled:
                    cv2.imshow("YOLO Person Detection", annotated_frame)
    
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
                # Update the last test time
                last_test_time = current_time
        except Exception as e:
            print(f"An error occured in loop, continuing...\n{e}")

except KeyboardInterrupt:
    print("\nStopping...")


# Release resources
webcam.release()
cv2.destroyAllWindows()
