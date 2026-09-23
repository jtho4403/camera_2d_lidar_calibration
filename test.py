import sys

import cv2
import numpy as np

def extract_checkerboard_corners(image_path, pattern_size):
    """
    Finds and refines checkerboard corners in an image.
    
    :param image_path: Path to the PNG file.
    :param pattern_size: Tuple of (columns, rows) of *internal* corners.
                         (e.g., an 8x8 checkerboard has 7x7 internal corners)
    :return: image, corners
    """
    # 1. Load and convert the image to grayscale
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded. Check the path.")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Find internal chessboard corners
    # flags can be adjusted to optimize detection (e.g., adaptive thresholding)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        # 3. Refine corner locations to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 4. Draw and display the detected corners
        img_with_corners = img.copy()
        cv2.drawChessboardCorners(img_with_corners, pattern_size, corners_subpix, ret)
        
        return img_with_corners, corners_subpix
    else:
        print("Checkerboard pattern not found.")
        return img, None

# --- Configuration ---
# PNG file path, passed on the command line
image_file = sys.argv[1]

# The inner vertices count (Width, Height). 
# Example: If your board has 9x9 squares, the internal corners are 8x8.
# Must match checkerboard_width/_height in cam_intrinsic.py and cam_lidar_2d_icp.py.
grid_size = (6, 3) 

# --- Execution ---
annotated_image, detected_corners = extract_checkerboard_corners(image_file, grid_size)

if detected_corners is not None:
    print(f"Successfully extracted {len(detected_corners)} corners!")
    
    # Save the result
    cv2.imwrite('extracted_corners.png', annotated_image)
    
    # Display the result in a window (Press any key to close)
    cv2.imshow('Checkerboard Corners', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
