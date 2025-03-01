# Visual Recognition Assignment 1

## Overview
This project consists of two parts:

1. **Coin Detection and Counting:** Use computer vision techniques to detect, segment, and count coins from an
image containing scattered Indian coins.
2. **Image Stitching:** Create a stitched panorama from multiple overlapping images.

Both parts are implemented in Python.

---

## Part 1: Coin Detection and Counting

### Dependencies
To install the required dependencies, run:
```bash
pip install opencv-python numpy
```

### Description
The coin detection script processes images to:
- Convert them to grayscale.
- Apply Gaussian blur to reduce noise.
- Detect edges using Sobel, Laplacian, and Canny edge detection methods.
- Find and draw contours around detected coins.
- Segment and save isolated coins.
- Count the number of coins detected.

### Usage
Run the following command to execute the coin detection script:
```bash
cd Part-1
python coin_detector.py
```

The script processes all jpg/jpeg images in the `Input` directory. For each `image-name`, it creates a corresponding `image-name-results` folder in the `Output` directory. Within this folder, you will find:
- `isloated-coins`: Segmented outputs for each coin in the input image.
- `edge-detection`: Resultant images after using various edge detectors.
- `detected-coins.jpg` Outlined coins in the original image.
- `segmented-coins.jpg`: Segmented coins in the original image.


### Key Functions
- `preprocess_image(image_path)`: Converts an image to grayscale and applies Gaussian blur.
- `detect_edges_sobel(blurred_image)`: Detects edges using the Sobel operator.
- `detect_edges_laplacian(blurred_image)`: Detects edges using the Laplacian operator.
- `detect_edges_canny(blurred_image)`: Detects edges using the Canny edge detector.
- `find_and_draw_contours(edges, image)`: Finds and draws contours around detected edges.
- `segment_and_save_coins(image_path, output_dir)`: Segments individual coins and saves them.
- `count_coins_in_image(segmented_image_path)`: Counts the number of coins detected.
- `process_all_images(input_dir, output_dir)`: Processes all images in a directory and applies coin detection.

---

## Part 2: Image Stitching

### Dependencies
To install the required dependencies, run:
```bash
pip install opencv-python numpy glob2 shutil
```

### Description
The image stitching script processes images to:
- Detect and extract keypoints using the SIFT algorithm.
- Match keypoints between images using FLANN-based matching.
- Use OpenCV's built-in `Stitcher` to merge images.
- Perform manual stitching with homography if automatic stitching fails.
- Crop black borders from the final stitched panorama.

### Usage
Run the following command to execute the image stitching script:
```bash
cd Part-2
python image_stitcher.py
```

The program will iterate over each folder in the `Input` directory. Each folder is expected to have overlapping images in jpg/jpeg format. For each folder, a folder in the `Output` directory will be created, which will contains three types of images:
For each folder, three types of images are generated:
- Keypoints present within an image.
- Images matching the keypoints between contiguous blocks.
- Intermediate image after stitching 2 images together, 3 images together and so on, until a panorama is obtained.
### Key Functions  

- `crop_black_borders(image)`: Removes black borders from the final stitched image to improve aesthetics.  
- `draw_and_save_keypoints(images, image_names, output_dir)`: Detects SIFT keypoints, visualizes them with crosses, and saves the modified images.  
- `draw_matches(img1, kp1, img2, kp2, matches, mask=None)`: Draws feature matches between two images for visualization.  
- `stitch_images(images, image_names, output_dir)`: Sequentially stitches images using feature matching, homography estimation, and blending.  \
- `process_image_sets(input_dir, output_dir, max_width=1200)`: Processes multiple image sets, extracts keypoints, and generates stitched panoramas.
---

## Directory Structure
```
Visual-Recognition-Assignment/
|-- Part-1
  |-- Input
  |-- Output
  |-- coin-detector.py
|-- Part-2
  |-- Input
  |-- Output
  |-- image-stitcher.py
│-- README.md
|-- Report.pdf
```

---

## Results
- The`Output` folders of `Part-1` and `Part-2` contain the results of the tasks performed respectively.

---

