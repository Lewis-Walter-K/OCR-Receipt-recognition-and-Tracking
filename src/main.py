import cv2
import matplotlib.pyplot as plt
import os

# Import the functions we just wrote
from segment import get_receipt_contour
from transform import process_and_flatten

def run_pipeline(image_path):
    # Create output folder if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    print("1. Running YOLO Segmentation...")
    points, original_image, annotated_image = get_receipt_contour(image_path, model_path="models/best.pt")
    
    if points is not None:
        print("2. Mathematical Flattening...")
        # Now returns two versions of the receipt
        flat_color, flat_bw = process_and_flatten(original_image, points)       

        # Save both results locally to output folder
        cv2.imwrite("output/receipt_flat_color.jpg", flat_color)
        cv2.imwrite("output/receipt_ocr_ready.jpg", flat_bw)
        print("Success! Saved 'output/receipt_flat_color.jpg' and 'output/receipt_ocr_ready.jpg'")
        
        # Plot the comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6)) 
        
        axes[0].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("YOLO Segmentation Mask")
        axes[0].axis("off")
        
        axes[1].imshow(cv2.cvtColor(flat_color, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Straightened Color Output")
        axes[1].axis("off")
        
        # B&W is single channel, plot directly as grayscale 
        axes[2].imshow(flat_bw, cmap="gray")
        axes[2].set_title("B&W Filter (Optimized for OCR)")
        axes[2].axis("off")
        
        plt.show()

if __name__ == "__main__":
    # Point this to your test receipt!
    run_pipeline("pic/receipt1.jpg")