import cv2
import numpy as np

def order_points(pts):
    """
    Sorts the 4 coordinates into a consistent order:
    [top-left, top-right, bottom-right, bottom-left]
    """
    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def flatten_receipt(image, pts):
    """
    Takes an image and the 4 corners of the receipt, and returns a flattened, top-down view.
    """
    # 1. Order the points consistently
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 2. Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 3. Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 4. Construct the destination points to map the screen to a top-down view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 5. Calculate the perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

    return warped

def process_and_flatten(image, yolo_polygon_points):
    """
    Main function to simplify YOLO points to 4 corners and flatten.
    """
    # Convert YOLO float coordinates to integers for OpenCV
    contour = np.array(yolo_polygon_points, dtype=np.int32)
    
    # Use OpenCV to find the bounding rectangle of the YOLO mask
    # This is a safe fallback in case YOLO draws a slightly rounded mask
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = box.astype(int) # These are our 4 corners!

    # Flatten the image using those 4 corners
    flat_receipt = flatten_receipt(image, box)
    
    return flat_receipt