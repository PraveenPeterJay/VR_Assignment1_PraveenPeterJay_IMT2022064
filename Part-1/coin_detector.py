import cv2
import numpy as np
import os
import shutil
import glob

# Preprocess image: Convert to grayscale and apply Gaussian blur
def preprocess_image(image_path):
    """
    Load an image, convert it to grayscale, and apply Gaussian blur for noise reduction.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.2)
    return image, gray, blurred

# Apply Sobel edge detection
def detect_edges_sobel(blurred_image):
    """
    Detect edges in the image using the Sobel operator.
    """
    blurred_image = cv2.GaussianBlur(blurred_image, (5, 5), 1.0)
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.convertScaleAbs(sobel_x) + cv2.convertScaleAbs(sobel_y)
    _, thresholded = cv2.threshold(sobel_combined, 100, 255, cv2.THRESH_BINARY)
    return thresholded

# Apply Laplacian edge detection
def detect_edges_laplacian(blurred_image):
    """
    Detect edges in the image using the Laplacian operator.
    """
    blurred_image = cv2.GaussianBlur(blurred_image, (5, 5), 1.0)
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F, ksize=3)
    return cv2.convertScaleAbs(laplacian)

# Apply Canny edge detection
def detect_edges_canny(blurred_image):
    """
    Detect edges in the image using the Canny edge detection algorithm.
    """
    return cv2.Canny(blurred_image, 50, 150)

# Detect and draw contours on the image
def find_and_draw_contours(edges, image):
    """
    Find contours in the edge-detected image and draw them on the original image.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = image.copy()
    cv2.drawContours(output_image, contours, -1, (255, 255, 0), 2)
    return output_image, contours

# Save processed images to disk
def save_image(image, output_dir, filename):
    """
    Save an image to the specified output directory with the given filename.
    """
    cv2.imwrite(os.path.join(output_dir, f"{filename}.jpg"), image)

# Segment coins and save cropped images
def segment_and_save_coins(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return
    
    # Prepare output directory
    coin_output_dir = os.path.join(output_dir, "isolated-coins")
    shutil.rmtree(coin_output_dir, ignore_errors=True)
    os.makedirs(coin_output_dir, exist_ok=True)
    
    # Convert to grayscale and apply preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    
    # Find contours of potential coins
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masked_image = np.zeros_like(image)
    coin_count = 1
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5000 < area < 35000:  # Ensure reasonable coin size
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
            if circularity > 0.7:  # More circular objects
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center, radius = (int(x), int(y)), int(radius)
                
                # Create mask for individual coin
                mask = np.zeros_like(gray)
                cv2.circle(mask, center, radius, 255, thickness=cv2.FILLED)
                isolated_coin = cv2.bitwise_and(image, image, mask=mask)
                
                # Crop coin region and save
                cropped_coin = isolated_coin[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]
                save_image(cropped_coin, coin_output_dir, f"coin-{coin_count}")  # Save as JPEG
                
                # Mark detected coins on output image
                cv2.circle(masked_image, center, radius, (255, 255, 0), thickness=-1)
                coin_count += 1
    
    # Save the segmented coins image
    save_image(masked_image, output_dir, "segmented-coins")  # Save as JPEG
    print(f"-> Segmented coins saved in '{coin_output_dir}'")

# Count the number of coins in the segmented image
def count_coins_in_image(segmented_image_path):
    """
    Count the number of coins in the segmented image.
    """
    image = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found!")
        return 0
    
    # Preprocess the image for counting
    blurred = cv2.GaussianBlur(image, (7, 7), 1.2)
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if 3000 < cv2.contourArea(cnt) < 40000]
    print(f"-> Number of coins detected: {len(valid_contours)}")
    return len(valid_contours)

# Process all images in the input directory
def process_all_images(input_dir, output_dir):
    """
    Process all images in the input directory, apply edge detection, segment coins, and save results.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.jpeg")) + glob.glob(os.path.join(input_dir, "*.jpg")))
    
    for file in image_files:
        print(f"Processing {os.path.basename(file)}...")
        image_name = os.path.splitext(os.path.basename(file))[0]
        set_output_path = os.path.join(output_dir, f"{image_name}-results")
        os.makedirs(set_output_path, exist_ok=True)
        
        # Preprocess the image
        image, gray, blurred = preprocess_image(file)
        
        # Apply edge detection
        canny_edges = detect_edges_canny(blurred)
        sobel_edges = detect_edges_sobel(gray)
        laplacian_edges = detect_edges_laplacian(gray)
        detected_image, _ = find_and_draw_contours(canny_edges, image)
        
        # Save edge detection results
        edge_output_path = os.path.join(set_output_path, "edge-detection")
        os.makedirs(edge_output_path, exist_ok=True)
        save_image(sobel_edges, edge_output_path, "sobel-edges")
        save_image(laplacian_edges, edge_output_path, "laplacian-edges")
        save_image(canny_edges, edge_output_path, "canny-edges")
        print(f"-> Edge detection results stored in {edge_output_path}")

        # Save coin detection result
        save_image(detected_image, set_output_path, "detected-coins")

        # Segment and save coins
        segment_and_save_coins(file, set_output_path)

        # Count the number of coins in the segmented coins images
        count_coins_in_image(os.path.join(set_output_path, "segmented-coins.jpg"))

        print()

# Execute coin detection on all images in the input directory
INPUT_DIRECTORY = "Input"
OUTPUT_DIRECTORY = "Output"
print()
process_all_images(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
