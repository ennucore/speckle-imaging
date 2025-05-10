#%% Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
import os

#%% Print available cameras
def list_available_cameras():
    """List all available camera devices."""
    available_cameras = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        ret, _ = cap.read()
        if ret:
            camera_name = f"Camera {index}"
            available_cameras.append((index, camera_name))
        cap.release()
        index += 1
    return available_cameras

available_cameras = list_available_cameras()
print(f"Found {len(available_cameras)} camera(s):")
for idx, name in available_cameras:
    print(f"  {idx}: {name}")

#%% Select camera and preview
def preview_camera(camera_idx=None):
    """Preview the camera feed from the specified camera index."""
    # If no camera index is provided, use the last one
    if camera_idx is None and available_cameras:
        camera_idx = available_cameras[-1][0]
    elif camera_idx is None:
        print("No cameras available.")
        return None, None
    
    # Open the camera
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_idx}")
        return None, None
    
    # Set higher resolution if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to capture image from camera {camera_idx}")
        cap.release()
        return None, None
    
    # Display the image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(rgb_frame)
    plt.title(f"Preview from Camera {camera_idx}")
    plt.axis('off')
    plt.show()
    
    return cap, frame

# Get the last camera by default
if available_cameras:
    default_camera_idx = available_cameras[1][0]
    print(f"Using camera {default_camera_idx} by default")
    cap, preview_frame = preview_camera(default_camera_idx)
else:
    print("No cameras detected")
    cap, preview_frame = None, None

# %%
import time
time.sleep(1)
# Play a beep sound to indicate completion or alert the user
import os
# Use system beep on macOS
os.system('afplay /System/Library/Sounds/Ping.aiff')



#%% Take first batch of images
def capture_image_batch(cap, num_images=5):
    """Capture a batch of images as quickly as possible."""
    if cap is None or not cap.isOpened():
        print("Camera not available")
        return None
    
    images = []
    for i in range(num_images):
        ret, frame = cap.read()
        if ret:
            images.append(frame)
        else:
            print(f"Failed to capture image {i+1}")
    
    print(f"Captured {len(images)} images")
    return images

# Capture first batch
print("Capturing first batch of images...")
batch1 = capture_image_batch(cap)

# Display the first image from the batch
if batch1 and len(batch1) > 0:
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(batch1[0], cv2.COLOR_BGR2RGB))
    plt.title("First image from Batch 1")
    plt.axis('off')
    plt.show()
# %%
os.system('afplay /System/Library/Sounds/Ping.aiff')
time.sleep(5)
os.system('afplay /System/Library/Sounds/Ping.aiff')

#%% Take second batch of images
print("Capturing second batch of images...")
batch2 = capture_image_batch(cap)

# Display the first image from the batch
if batch2 and len(batch2) > 0:
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(batch2[0], cv2.COLOR_BGR2RGB))
    plt.title("First image from Batch 2")
    plt.axis('off')
    plt.show()

#%% Subtract average second image from average first image
def compute_average_image(batch):
    """Compute the average image from a batch of images."""
    if not batch or len(batch) == 0:
        return None
    
    # Convert to float32 for better precision during averaging
    float_batch = [img.astype(np.float32) for img in batch]
    avg_img = np.mean(float_batch, axis=0)
    
    # Convert back to uint8 for display
    return avg_img.astype(np.uint8)

# %%
batch3 = capture_image_batch(cap)
# %%
def plot_normalized_difference(img1, img2, normalize_intensity=True, title_postfix=""):
    if isinstance(img1, list):
        img1 = compute_average_image(img1)
    if isinstance(img2, list):
        img2 = compute_average_image(img2)
    if normalize_intensity:
        img1 = img1 / np.mean(img1)
        img2 = img2 / np.mean(img2)

    diff_img = cv2.absdiff(img1, img2)
    log_diff = np.log1p(diff_img[:, :].astype(np.float32))
    log_diff = cv2.normalize(log_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    plt.imshow(cv2.cvtColor(log_diff, cv2.COLOR_BGR2RGB))
    plt.title(f"Difference {title_postfix}")
    plt.axis('off')


# %%
plot_normalized_difference(batch2, batch3, title_postfix="Batch 2 - Batch 3")

# %%
plot_normalized_difference(batch1, batch2, title_postfix="Batch 1 - Batch 2")



# %%