import cv2
import matplotlib.pyplot as plt

# Import the functions we just wrote
from segment import get_receipt_contour
from transform import process_and_flatten

def run_pipeline(image_path):
    print("1. Running YOLO Segmentation...")
    # Call your working segment function
    points, original_image, annotated_image = get_receipt_contour(image_path, model_path="models/best.pt")
    
    if points is not None:
        print("2. Mathematical Flattening...")
        # Pass the original image and the YOLO points to our new transform function
        flat_image = process_and_flatten(original_image, points)
        
        # Save the beautiful, scanner-like output!
        cv2.imwrite("final_scanned_receipt.jpg", flat_image)
        print("Success! Saved as 'final_scanned_receipt.jpg'")
        
        # Let's view both side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Convert BGR to RGB for correct colors in matplotlib
        img_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        flat_rgb = cv2.cvtColor(flat_image, cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(img_rgb)
        axes[0].set_title("YOLO Mask")
        axes[0].axis("off")
        
        axes[1].imshow(flat_rgb)
        axes[1].set_title("Flattened Output (Ready for OCR)")
        axes[1].axis("off")
        
        plt.show()

if __name__ == "__main__":
    # Point this to your test receipt!
    run_pipeline("pic/receipt1.jpg")