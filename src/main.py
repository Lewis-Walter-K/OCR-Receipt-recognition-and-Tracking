import os
import torch
# Throttle threads immediately at runtime startup to avoid VS Code Terminal freezing
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

import cv2
import matplotlib.pyplot as plt
import json

# Import custom pipelines
from segment import get_receipt_contour
from transform import process_and_flatten
from ocr import initialize_ocr, format_extracted_text, fix_orientation

def run_pipeline(image_path, country):
    print("\n====================================")
    print("      INITIALIZING PIPELINE         ")
    print("====================================")
    
    # Initialize Core Engine
    ocr_engine = initialize_ocr(country)
    
    # Step 1: YOLO Segmentation
    print("\nExecuting Step 1: YOLO Segmentation Extraction...")
    points, original_image, annotated_image = get_receipt_contour(image_path, model_path="models/best.pt")
    
    if points is not None:
        # Step 2: Transformation & Black and White Conversion
        print("Executing Step 2: Running 4-Corner Geometric Straightening...")
        flat_color, flat_bw = process_and_flatten(original_image, points)
        
        # Step 3: Orientation Check (Using  Black & White image)
        print("Executing Step 3: Assessing Text Density & Orientation Flow...")
        final_upright_bw, raw_ocr_results = fix_orientation(ocr_engine, flat_bw)
        
        os.makedirs("output/final_pic", exist_ok=True)

        # Step 4: Text Extraction & Formatting
        print("Executing Step 4: Formatting OCR Data...")
        receipt_lines = format_extracted_text(raw_ocr_results)

        # --- NEW STEP 5: THE JSON CHECKPOINT ---
        print("Executing Step 5: Exporting NLP Checkpoint...")
        json_path = "output/receipt_data.json"

        # Save the dictionary list to a JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(receipt_lines, f, ensure_ascii=False, indent=4)

        # Export assets 
        cv2.imwrite("output/final_pic/receipt_final_upright.jpg", final_upright_bw)
        print("\nProcessed system files saved successfully to your disk workspace!")
        
        # Printing results to console stream
        print("\n====================================")
        print("     UPRIGHT DIGITAL TRANSCRIPTION   ")
        print("====================================")
        for item in receipt_lines:
            if item['confidence'] > 0.10:  # Keeps genuine strings, filters noise
                print(f"{item['text']}  (Conf: {item['confidence']:.2f})")
        print("====================================\n")
        
        # # Visual Plotting Pipeline Stream Check
        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # axes[0].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        # axes[0].set_title("1. YOLO Input Target Frame")
        # axes[0].axis("off")
        
        # axes[1].imshow(final_upright_bw, cmap="gray")
        # axes[1].set_title("2. Adjusted Final Upright OCR Matrix")
        # axes[1].axis("off")
        # plt.show()
        
    else:
        print("\nCritical Failure: Target boundary mask mapping failed.")

if __name__ == "__main__":
    # Scanning a Vietnamese or European receipt
    run_pipeline("pic/receipt1.jpg", country="VN")
    
    # Scanning a Japanese receipt
    # run_pipeline("pic/tokyo_store.jpg", country="JP")
    
    # Scanning an Arabic receipt
    # run_pipeline("pic/dubai_mall.jpg", country="AE")