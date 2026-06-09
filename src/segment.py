import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

def get_receipt_contour(image_path, model_path="models/best.pt"):
    # 1. Load your custom trained model weights
    model = YOLO(model_path)
    
    # 2. Run inference on a local image
    results = model.predict(source=image_path, conf=0.25, save=False)
    result = results[0]
    
    # 3. Extract the polygon points of the segmentation mask
    if result.masks is not None:
        polygon_points = result.masks.xy[0] 
        
        # NEW: Generate the visual image with YOLO's built-in drawing tool
        annotated_image = result.plot() 
        
        return polygon_points, result.orig_img, annotated_image
    else:
        print("No receipt detected in the image.")
        return None, None, None

# --- Quick Test ---
if __name__ == "__main__":
    # Create output folder if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Test it on your raw smartphone photo
    points, original_image, annotated_image = get_receipt_contour("pic/receipt1.jpg")
    
    if points is not None:
        print(f"Successfully found the receipt! Outline contains {len(points)} points.")
        
        # --- TWO WAYS TO VIEW THE RESULT ---
        
        # Method 1: Save it as a new file in your folder (Foolproof)
        cv2.imwrite("output/output_receipt.jpg", annotated_image)
        print("Saved visual output to 'output/output_receipt.jpg'. Open your folder to view it!")
        
        # Method 2: Pop it up on your screen right now (using Matplotlib)
        # We have to convert the colors from OpenCV's BGR to standard RGB first
        display_img = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 8))
        plt.imshow(display_img)
        plt.title("YOLO Segmentation Output")
        plt.axis('off')
        plt.show()