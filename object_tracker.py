import cv2
import time
from ultralytics import YOLO

# 1. Auto-create the custom tracker with all required settings
with open('custom_tracker.yaml', 'w') as f:
    f.write("""tracker_type: bytetrack
track_high_thresh: 0.25 
track_low_thresh: 0.1   
new_track_thresh: 0.25  
track_buffer: 90        
match_thresh: 0.8
gmc_method: sparseOptFlow
fuse_score: True""")

# 2. Load the upgraded YOLO medium model
model = YOLO('yolov8m.pt')

# --- NEW: Interactive Terminal Prompt ---
print("\n" + "="*45)
print(" CodeAlpha Object Tracker - Select Source ")
print("="*45)
user_input = input("Type a video filename (e.g. test_video.mp4) OR press ENTER for Live Webcam: ").strip()

if user_input == "":
    video_source = 0
    output_name = 'live_output.mp4'
    print("-> Connecting to Webcam...")
else:
    video_source = user_input
    output_name = f"tracked_{user_input}"
    print(f"-> Loading file: {video_source}...")

# 3. Open the chosen video source
video = cv2.VideoCapture(video_source)

# 4. Safety check to ensure the file or webcam actually opened
if not video.isOpened():
    print(f"\nERROR: Could not open '{video_source}'. Please check if the file exists and try again.")
    exit()

# 5. Get video properties (Defaulting to 30 FPS if a webcam reports 0)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
if fps <= 0: 
    fps = 30 

# 6. Set up the VideoWriter to save the session dynamically based on your choice
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

# Feature Variables
unique_objects = set() 
prev_time = 0          

print("\nProcessing started... Press 'q' in the video window to stop and save.")

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("Processing complete!")
        break 
        
    # 7. Detect and Track objects
    results = model.track(frame, persist=True, tracker="custom_tracker.yaml", conf=0.15, classes=[0, 2, 3, 5, 7], verbose=False)
    
    annotated_frame = results[0].plot()
    
    # 8. Extract IDs and update the unique counter
    if results[0].boxes.id is not None:
        current_ids = results[0].boxes.id.cpu().numpy().astype(int)
        unique_objects.update(current_ids)
        
    # 9. Calculate FPS safely
    curr_time = time.time()
    time_diff = curr_time - prev_time
    fps_display = 1 / time_diff if time_diff > 0 else 0
    prev_time = curr_time
    
    # 10. Draw UI overlays
    cv2.putText(annotated_frame, f"FPS: {int(fps_display)}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Total Unique Objects: {len(unique_objects)}", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # 11. Write and Display
    out.write(annotated_frame)
    cv2.imshow("CodeAlpha Object Tracker", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Processing stopped by user.")
        break

# 12. Clean up
video.release()
out.release() 
cv2.destroyAllWindows()
print(f"Saved successfully as: {output_name}")