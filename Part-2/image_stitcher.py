import os
import cv2
import glob
import shutil
import numpy as np

def crop_black_borders(image):
    """Removes black borders from a stitched image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # Create a binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))  # Get bounding box of largest contour
        return image[y:y+h, x:x+w]  # Crop the image
    return image

def draw_and_save_keypoints(images, image_names, output_dir):
    """Detects and draws SIFT keypoints on images, saving them in the output directory."""
    sift = cv2.SIFT_create()
    keypoints_descriptors = []
    
    for i, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        keypoints, descriptors = sift.detectAndCompute(gray, None)  # Detect keypoints and compute descriptors
        keypoints_descriptors.append((keypoints, descriptors))
        
        # Draw keypoints on the image
        keypoint_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), 
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(os.path.join(output_dir, f"keypoints-{image_names[i]}.jpg"), keypoint_image)  # Save keypoint image
    
    return keypoints_descriptors

def stitch_images(images, image_names, output_dir):
    """Attempts to stitch images using OpenCV's built-in Stitcher; falls back to manual stitching if needed."""
    keypoints_descriptors = draw_and_save_keypoints(images, image_names, output_dir)
    
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, panorama = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        return crop_black_borders(panorama)  # Crop stitched image to remove black borders
    
    print(f"Built-in stitching failed with error code: {status}. Falling back to manual stitching...")
    return manual_stitch(images, keypoints_descriptors)

def manual_stitch(images, keypoints_descriptors):
    """Manually stitches images using feature matching and homography estimation."""
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    max_height = max(img.shape[0] for img in images)
    max_width = sum(img.shape[1] for img in images)
    
    # Create a blank canvas large enough to hold all stitched images
    canvas = np.zeros((max_height * 2, max_width * 2, 3), dtype=np.uint8)
    center_index = len(images) // 2  # Choose the center image as reference
    base_image = images[center_index]
    
    # Place the base image in the middle of the canvas
    start_x, start_y = max_width // 4, max_height // 4
    canvas[start_y:start_y + base_image.shape[0], start_x:start_x + base_image.shape[1]] = base_image
    mask = np.zeros_like(canvas, dtype=np.uint8)  # Initialize mask for blending
    mask[start_y:start_y + base_image.shape[0], start_x:start_x + base_image.shape[1]] = 255
    
    for i in range(len(images)):
        if i == center_index:
            continue  # Skip the reference image
        
        kp1, des1 = keypoints_descriptors[center_index]
        kp2, des2 = keypoints_descriptors[i]
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            print(f"Not enough descriptors for image {i}.")
            continue
        
        # Match features using FLANN-based matcher
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]  # Apply Lowe's ratio test
        
        if len(good_matches) < 4:
            print(f"Not enough matches between images {center_index} and {i}.")
            continue
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Compute homography matrix
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            print(f"Homography not found for image {i}.")
            continue
        
        # Warp the image and merge it with the canvas
        warped_image = cv2.warpPerspective(images[i], H, (canvas.shape[1], canvas.shape[0]))
        canvas = np.where(warped_image > 0, warped_image, canvas)
    
    return crop_black_borders(canvas)

def process_image_sets(input_dir, output_dir, max_width=1200):
    """Processes all image sets in the input directory and saves stitched panoramas in the output directory."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Clear existing output directory
    os.makedirs(output_dir)
    
    image_sets = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for set_name in image_sets:
        set_input_path = os.path.join(input_dir, set_name)
        set_output_path = os.path.join(output_dir, set_name)
        os.makedirs(set_output_path, exist_ok=True)
        
        # Load images from the current set
        image_files = sorted(glob.glob(os.path.join(set_input_path, "*.jpg")))
        images, image_names = [], []
        
        for file in image_files:
            image_name = os.path.splitext(os.path.basename(file))[0]  # Get image name without extension
            image = cv2.imread(file)
            
            if image is not None:
                image_names.append(image_name)
                images.append(image)
            else:
                print(f"Warning: Could not load {file}")
        
        if len(images) < 2:
            print(f"Skipping {set_name}: Not enough images to stitch.")
            continue
        
        # Resize images if they exceed max width while maintaining aspect ratio
        images = [cv2.resize(img, (int(max_width), int(img.shape[0] * (max_width / img.shape[1])))) if img.shape[1] > max_width else img for img in images]
        panorama = stitch_images(images, image_names, set_output_path)
        cv2.imwrite(os.path.join(set_output_path, "stitched-panorama.jpg"), panorama)
        print(f"Results for {set_input_path} stored in {set_output_path}")

# Begin execution of image stitching
INPUT_DIRECTORY = "Input"
OUTPUT_DIRECTORY = "Output"
process_image_sets(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
