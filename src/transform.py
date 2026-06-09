import cv2
import numpy as np

def order_points(pts):
    """
    Sorts 4 coordinates into a consistent order:
    [top-left, top-right, bottom-right, bottom-left]
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left has the smallest sum, bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has the smallest difference, bottom-left has the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def get_exact_four_corners(contour):
    """
    Uses polygon approximation to find the true 4 corners of the receipt mask.
    Falls back to extreme points if the mask shape is complex.
    """
    # Simplify the contour shape to eliminate tiny ripples along the edge
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    # If our simplified polygon has exactly 4 points, we found our corners!
    if len(approx) == 4:
        return approx.reshape(4, 2)
    
    # Fallback: If approximation gives more/fewer points, find the 4 absolute extremes
    pts = contour.reshape(-1, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # top-left
    rect[2] = pts[np.argmax(s)] # bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    
    return rect

def flatten_receipt(image, pts):
    """
    Warps the perspective of the image to make it top-down and perfectly straight.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate maximum width
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Calculate maximum height
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Target destination canvas
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))
    
    return warped

def optimize_for_ocr(warped_image):
    """
    Converts a color cropped receipt into a high-contrast black and white image,
    removing shadows and artifacts to maximize OCR accuracy.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Adaptive Gaussian Thresholding
    # This evaluates pixels in small local neighborhoods (e.g., 21x21 blocks).
    # It dynamically removes localized shadows while keeping text sharp.
    ocr_ready = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        21, # Neighborhood block size (must be odd)
        10  # Constant subtracted from the mean
    )
    
    return ocr_ready

def process_and_flatten(image, yolo_polygon_points):
    """
    Main orchestration function.
    """
    contour = np.array(yolo_polygon_points, dtype=np.int32)
    
    # FIX: Get true geometric corners instead of a minimum bounding box
    corners = get_exact_four_corners(contour)
    
    # Flatten perspective
    flat_receipt = flatten_receipt(image, corners)
    
    # Optimize text contrast for OCR
    ocr_version = optimize_for_ocr(flat_receipt)
    
    return flat_receipt, ocr_version