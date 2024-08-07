#!pip install ultralytics
import ultralytics
ultralytics.__version__

#!pip install supervision
import supervision
print("supervision.__version__:", supervision.__version__)

#!pip install torch
import torch
torch.__version__

#detection
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on 'bus.jpg' with arguments
model.predict(source="/content/video.mp4", save=True, imgsz=320, conf=0.5)

#tracking
model = YOLO('yolov8n.pt')

results = model.track(source="/content/video.mp4",conf=0.3, iou=0.5, save=True, tracker="bytetrack.yaml")
import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Set up video capture
cap = cv2.VideoCapture("/content/video.mp4")

# Define the line coordinates
START_init = sv.Point(130, 79)
END_init = sv.Point(600, 74)

START_final = sv.Point(10, 400)
END_final = sv.Point(880, 400)


# Store the track history
track_history = defaultdict(lambda: [])

# Create a dictionary to keep track of objects that have crossed the line
crossed_objects = {}

#dictionary for final line
crossed_objects_final = {}

# Open a video sink for the output video
video_info = sv.VideoInfo.from_video_path("/content/video.mp4")
with sv.VideoSink("output_2line.mp4", video_info) as sink:

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, classes=[2, 3, 5, 7], persist=True, save=True, tracker="bytetrack.yaml")

            num_cars_current_frame = len(results[0].boxes)
            print("Number of cars in current frame:", num_cars_current_frame)
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            # detections = sv.Detections.from_yolov8(results[0])

            # Plot the tracks and count objects crossing the line
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)

                # Check if the object crosses the line
                if START_init.x < x < END_init.x and abs(y - START_init.y) < 1:  # Assuming objects cross horizontally
                    if track_id not in crossed_objects:
                        crossed_objects[track_id] = True

                if START_final.x < x < END_final.x and abs(y - START_final.y) < 1:  # Assuming objects cross horizontally
                    if track_id not in crossed_objects_final:
                        crossed_objects_final[track_id] = True

                    # Annotate the object as it crosses the line
                cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

            # Draw the line on the frame
            cv2.line(annotated_frame, (START_init.x, START_init.y), (END_init.x, END_init.y), (0, 255, 0), 2)
            #Draw the final line on the frame
            cv2.line(annotated_frame, (START_final.x, START_final.y), (END_final.x, END_final.y), (0, 255, 0), 2)
            # Write the count of objects on each frame
            count_text_init = f"Objects crossed initial: {len(crossed_objects)}"
            count_text_final = f"Objects crossed final: {len(crossed_objects_final)}"
            Ncars_init = len(crossed_objects)
            Ncars_final = len(crossed_objects_final)
            total_cars_waiting = num_cars_current_frame - (abs(Ncars_init - Ncars_final))
            # total_cars_waiting = Ncars_final - Ncars_init
            Tmax = 180
            Tmin = 30
            #tc is average time taken by one car to pass the trafic light
            tc = 5
            #Tred is duration of red light
            Tred = min(max((total_cars_waiting * tc), Tmin), Tmax)
            tred_text = f"Time Calculated For the Red Light: {Tred} seconds"
            num_cars_current_frame_text = f"Number of cars in current frame: {num_cars_current_frame}"
            diff_cars_text = f"Number of cars waiting: {abs(Ncars_init - Ncars_final)}"
            # cv2.putText(annotated_frame, count_text_init, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, diff_cars_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, tred_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, num_cars_current_frame_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Write the frame with annotations to the output video
            sink.write_frame(annotated_frame)
        else:
            break

# Release the video capture
cap.release()

total_cars_waiting = Ncars_init - Ncars_final
Tmax = 180
Tmin = 30
#tc is average time taken by one car to pass the trafic light
tc = 2
#basic formula
#Tred is duration of red light
print(Ncars_init)
print(Ncars_final)
print(Ncars_init-Ncars_final)
Tred = total_cars_waiting * tc
print(Tred)

#TO ensure the Tred stays within reasonable limit
 Tred = min(max((total_cars_waiting * tc), Tmin), Tmax)
 print(Tred)
