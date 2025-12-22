import cv2
import os

# Parameters
image_folder = "/data/point_cloud/video/pufm_wo_pa_vis"  # Change this to your image folder
output_video = "/data/point_cloud/video/pufm_wo_pa_vis/ddpm.mp4"
fps = 2  # Adjust the frame rate as needed
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 0, 0)  # White color
thickness = 2
position = (10, 30)  # Top-left corner

# Get list of images
images = [f"m333_{str(i).zfill(2)}.jpg" for i in range(99, 0, -1)]  # Ensure the correct naming pattern
images = [os.path.join(image_folder, img) for img in images if os.path.exists(os.path.join(image_folder, img))]

# Read the first image to get dimensions
frame = cv2.imread(images[0])
h, w, _ = frame.shape

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4
video = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

# Process images
for i, img_path in enumerate(images, start=1):
    frame = cv2.imread(img_path)
    if frame is None:
        continue
    
    # Add timestep text
    cv2.putText(frame, f"t = {i}", position, font, font_scale, font_color, thickness)
    
    # Write to video
    video.write(frame)

# Release video
video.release()
cv2.destroyAllWindows()

print("Video saved as", output_video)
