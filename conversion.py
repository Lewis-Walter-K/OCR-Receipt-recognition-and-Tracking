import json
import os
import shutil

def convert_and_reorganize_dataset(base_dir):
    # The three folders in your dataset
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        json_path = os.path.join(split_dir, '_annotations.coco.json')
        if not os.path.exists(json_path):
            print(f"No JSON found in {split}, skipping.")
            continue

        print(f"Processing {split} folder...")

        # 1. Create the YOLO format subdirectories
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # 2. Load the COCO JSON
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
            
        # Create a dictionary to quickly look up image data
        images_info = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by their image_id
        annotations_by_image = {}
        for ann in coco_data.get('annotations', []):
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # 3. Process every image listed in the JSON
        for img_id, img_info in images_info.items():
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Move the image file to the new /images/ subdirectory
            src_image_path = os.path.join(split_dir, img_filename)
            dst_image_path = os.path.join(images_dir, img_filename)
            
            if os.path.exists(src_image_path):
                shutil.move(src_image_path, dst_image_path)
            
            # Prepare the corresponding .txt label file
            txt_filename = os.path.splitext(img_filename)[0] + '.txt'
            txt_filepath = os.path.join(labels_dir, txt_filename)
            
            # 4. Normalize coordinates and write to .txt
            with open(txt_filepath, 'w') as txt_file:
                if img_id in annotations_by_image:
                    for ann in annotations_by_image[img_id]:
                        # Your JSON shows 'receipt' is id 1. YOLO classes start at 0.
                        category_id = ann['category_id'] - 1 
                        
                        for seg in ann['segmentation']:
                            yolo_coords = []
                            for i in range(0, len(seg), 2):
                                x = seg[i] / img_width
                                y = seg[i+1] / img_height
                                yolo_coords.extend([f"{x:.5f}", f"{y:.5f}"])
                            
                            txt_file.write(f"{category_id} " + " ".join(yolo_coords) + "\n")
                            
    print("\nSuccess! Dataset is now in YOLOv8 format.")

# --- INSTRUCTIONS ---
# Replace the path below with the absolute path to the folder containing train, valid, and test
# Example: '/kaggle/input/your-dataset-name' or 'C:/Users/YourName/Downloads/dataset'
dataset_path = r'C:\Code\Year_2\Phase3\Intro_to_DS_AI\OCR-Receipt-recognition-and-Tracking\Receipt' 

convert_and_reorganize_dataset(dataset_path)