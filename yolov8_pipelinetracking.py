import cv2
from ultralytics import YOLO
import numpy as np
import time

# Load the YOLOv8 model
model = YOLO('/home/hasan/BelajarYOLO/YOLOV8_Pipelinetracking/pipelinev8n-seg.pt')

# Open the video file
video_path = "/home/hasan/BelajarYOLO/YOLOV8_Pipelinetracking/Pipeline_Youtube.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize variables
masked_frame = None
final_mask = None
fps_start_time = time.time()
fps = 0

# Main loop for processing video frames
while True:
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # resize the frame to match the model
        frame = cv2.resize(frame, (640, 384))
        # Run YOLOv8 inference on the frame
        results = model(frame, max_det=2)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Extract masks of the detected objects
        for r in results:
            if r.masks is not None:
                for j, masked_frame in enumerate(r.masks.data):
                    masked_frame = masked_frame.numpy() * 255
                    masked_frame = masked_frame.astype('uint8')

        # Find contours and draw largest contour on final_mask
        contours, _ = cv2.findContours(masked_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            final_mask = np.zeros_like(masked_frame)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Process final_mask to find upper and lower detection points, draw lines and circles
        if final_mask is not None:
            frame_width =  final_mask.shape[1]
            frame_height =  final_mask.shape[0]

            # Check the nearest y value of detected pipe from the top of the frame
            for i in range(frame_height):
                if np.any(final_mask[i, :] > 0):
                    y_upper_detect = i
                    break
            
            # Set upper detected point of the pipe
            line_indices = np.where(final_mask[y_upper_detect, :] > 0)
            x_upper_detect = int(np.mean(line_indices))
            upper_detect = (x_upper_detect, y_upper_detect)

            # Set lower detected point of the pipe
            line_indices = np.where(final_mask[frame_height - 10, :] > 0)
            if len(line_indices[0]) > 0:
                x_lower_detect = int(np.mean(line_indices))
                lower_detect = (x_lower_detect, frame_height - 10)
            else:
                lower_detect = (int(frame_width/2), frame_height - 10)

            # Draw dot and line for the detected pipe
            cv2.circle(frame, upper_detect, radius=3, color=(255, 255, 255), thickness=-1)
            cv2.circle(frame, lower_detect, radius=3, color=(255, 255, 255), thickness=-1)
            cv2.line(frame, lower_detect, upper_detect, color=(255, 0, 0), thickness=2)

            # Set the center target point for upper and lower frame
            upper_target = (int(frame_width/2), y_upper_detect)
            lower_target = (int(frame_width/2), frame_height - 10)

            cv2.circle(frame, upper_target, radius=4, color=(0, 0, 255), thickness=-1)
            cv2.circle(frame, lower_target, radius=4, color=(0, 0, 255), thickness=-1)

            cv2.line(frame, upper_detect, upper_target, color=(0, 0, 255), thickness=1)
            cv2.line(frame, lower_detect, lower_target, color=(0, 0, 255), thickness=1)

            # Offset pixel value that can use as error feedback for controller
            upper_offset = x_upper_detect - int(frame_width/2)
            lower_offset = x_lower_detect - int(frame_width/2)

            cv2.putText(frame, str(upper_offset), upper_target, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=1)
            cv2.putText(frame, str(lower_offset), lower_target, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=1)

        # Calculate FPS
        fps_end_time = time.time()
        fps_diff_time = fps_end_time - fps_start_time
        fps = 1 / fps_diff_time
        fps_start_time = fps_end_time

        # Display FPS information on the frame
        fps_text = "FPS:{:.2f}".format(fps)
        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

        # Show the processed frame with annotations
        cv2.imshow("Pipefollowing", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
