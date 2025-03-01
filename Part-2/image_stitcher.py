import os
import cv2
import glob
import shutil
import numpy as np

def remove_black_borders(image):
    """Crops out black borders from an image by detecting non-black regions."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y + h, x:x + w]
    return image

def detect_and_annotate_keypoints(images, filenames, output_folder):
    """Detects SIFT keypoints and annotates them with cross markers."""
    sift = cv2.SIFT_create()
    keypoints_data = []
    
    for idx, img in enumerate(images):
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(grayscale, None)
        keypoints_data.append((keypoints, descriptors))

        # Create copy to draw keypoints as crosses
        annotated_img = img.copy()
        for keypoint in keypoints:
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
            size = int(keypoint.size)
            color = (255, 255, 0)  # Cyan
            thickness = 1
            
            # Draw cross shape
            cv2.line(annotated_img, (x - size, y), (x + size, y), color, thickness)
            cv2.line(annotated_img, (x, y - size), (x, y + size), color, thickness)
        
        cv2.imwrite(os.path.join(output_folder, f"keypoints-{filenames[idx]}.jpg"), annotated_img)
    
    return keypoints_data

def visualize_feature_matches(image1, keypoints1, image2, keypoints2, matches, mask=None):
    """Generates an image displaying feature matches between two images."""
    return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
                           matchColor=(255, 255, 0), singlePointColor=(255, 0, 0),
                           matchesMask=mask, flags=cv2.DrawMatchesFlags_DEFAULT)

def stitch_images(images, keypoints_data, output_folder):
    """Stitches images together sequentially using keypoint matching."""
    if len(images) < 2:
        return images[0]
    
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    stitched_result = images[0]
    
    for i in range(1, len(images)):
        img_prev, img_next = stitched_result, images[i]
        
        # Compute keypoints for the current stitched image
        sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04)
        kp_prev, des_prev = sift.detectAndCompute(cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY), None)
        kp_next, des_next = keypoints_data[i]
        
        if des_prev is None or des_next is None or len(des_prev) < 4 or len(des_next) < 4:
            print(f"Skipping image {i+1} due to insufficient descriptors.")
            continue
        
        # Match features using KNN
        knn_matches = matcher.knnMatch(des_next, des_prev, k=2)
        good_matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]
        
        # Save match visualization
        match_vis = visualize_feature_matches(img_next, kp_next, img_prev, kp_prev, good_matches)
        cv2.imwrite(os.path.join(output_folder, f"matches-between-{i}-and-{i+1}.jpg"), match_vis)
        
        if len(good_matches) < 10:
            print(f"Skipping image {i+1} due to insufficient matches: {len(good_matches)}")
            continue
        
        # Extract matching points
        src_pts = np.float32([kp_next[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_prev[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Compute homography
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            print(f"Skipping image {i+1} due to failed homography calculation.")
            continue
        
        # Warp image and blend
        height, width = img_prev.shape[:2]
        panorama = cv2.warpPerspective(img_next, H, (width * 2, height))
        panorama[0:height, 0:width] = img_prev
        
        # Remove black borders
        stitched_result = remove_black_borders(panorama)
        cv2.imwrite(os.path.join(output_folder, f"intermediate-{i+1}.jpg"), stitched_result)
    
    return stitched_result

def process_image_folder(input_folder, output_folder, max_width=1200):
    """Processes image folders, detecting keypoints and stitching them into panoramas."""
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    image_collections = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    
    for collection in image_collections:
        collection_path = os.path.join(input_folder, collection)
        output_path = os.path.join(output_folder, collection)
        os.makedirs(output_path, exist_ok=True)
        
        image_files = sorted(glob.glob(os.path.join(collection_path, "*.jpeg")) + glob.glob(os.path.join(collection_path, "*.jpg")))
        images, filenames = [], []
        
        for file in image_files:
            image_name = os.path.splitext(os.path.basename(file))[0]
            image = cv2.imread(file)
            if image is not None:
                filenames.append(image_name)
                images.append(image)
            else:
                print(f"Warning: Skipping {file}, unable to load.")
        
        if len(images) < 2:
            print(f"Skipping {collection}: Not enough images for stitching.")
            continue
        
        # Resize images to maintain aspect ratio
        images = [cv2.resize(img, (max_width, int(img.shape[0] * (max_width / img.shape[1]))))
                  if img.shape[1] > max_width else img for img in images]
        
        # Detect keypoints and stitch
        keypoints_info = detect_and_annotate_keypoints(images, filenames, output_path)
        final_panorama = stitch_images(images, keypoints_info, output_path)
        cv2.imwrite(os.path.join(output_path, "stitched-panorama.jpg"), final_panorama)
        print(f"Saved results for {collection_path} in {output_path}")

# Executing the code
INPUT_DIR = "Input"
OUTPUT_DIR = "Output"
process_image_folder(INPUT_DIR, OUTPUT_DIR)
