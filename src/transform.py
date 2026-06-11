import cv2
import numpy as np

def order_points(pts):
    """Sorts 4 coordinates into order: [top-left, top-right, bottom-right, bottom-left]"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_exact_four_corners(contour):
    """Simplifies complex YOLO masks down to 4 distinct geometric corners."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)
    
    # Fallback extreme points
    pts = contour.reshape(-1, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def flatten_receipt(image, pts):
    """Performs perspective warp transformation to straighten the document."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

def optimize_for_ocr(warped_image):
    """Binarizes image to pure black and white to maximize OCR visibility."""
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )

def process_and_flatten(image, yolo_polygon_points):
    contour = np.array(yolo_polygon_points, dtype=np.int32)
    corners = get_exact_four_corners(contour)
    flat_receipt = flatten_receipt(image, corners)
    ocr_version = optimize_for_ocr(flat_receipt)
    return flat_receipt, ocr_version