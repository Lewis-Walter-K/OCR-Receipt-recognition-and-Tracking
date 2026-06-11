import easyocr

import cv2
import easyocr

def initialize_ocr(country_code="GLOBAL_LATIN"):
    """
    Dynamically initializes the optimal PyTorch OCR model based on the target region.
    Prevents memory crashes by loading only the necessary script matrix.
    """
    country_code = country_code.upper()
    
    if country_code in ["VN", "VIETNAM", "GLOBAL_LATIN", "US", "UK", "FR"]:
        print("Loading Engine: Latin/European Script Matrix (EN, VI, FR, ES)...")
        return easyocr.Reader(['vi', 'en'], gpu=False)
        
    elif country_code in ["CN", "CHINA", "CHINESE"]:
        print("Loading Engine: Simplified Chinese Character Matrix...")
        return easyocr.Reader(['ch_sim', 'en'], gpu=False)
        
    elif country_code in ["JP", "JAPAN", "JAPANESE"]:
        print("Loading Engine: Japanese Kanji/Kana Matrix...")
        return easyocr.Reader(['ja', 'en'], gpu=False)
        
    elif country_code in ["KR", "KOREA", "KOREAN"]:
        print("Loading Engine: Korean Hangul Matrix...")
        return easyocr.Reader(['ko', 'en'], gpu=False)
        
    elif country_code in ["AE", "ARABIC", "EGYPT", "SA"]:
        print("Loading Engine: Arabic Right-to-Left Cursive Matrix...")
        return easyocr.Reader(['ar', 'en'], gpu=False)
        
    else:
        print("Fallback Engine: Defaulting to Universal English...")
        return easyocr.Reader(['en'], gpu=False)
    
def fix_orientation(ocr_model, bw_image_array):
    """
    1. Checks orientation using standard reading (to access confidence scores).
    2. Once orientation is confirmed, extracts final text using tuned paragraph formatting.
    Returns: (raw_ocr_data, upright_image_array)
    """

    def get_confidence(img):
        results = ocr_model.readtext(img)
        total_conf = sum([line[2] for line in results]) if results else 0
        avg_conf = (total_conf / len(results)) if results else 0
        return avg_conf

    # 1. Check 0-degree orientation
    conf_0 = get_confidence(bw_image_array)
    
    # 2. Check 180-degree orientation
    flipped_image = cv2.rotate(bw_image_array, cv2.ROTATE_180)
    conf_180 = get_confidence(flipped_image)
    
    # 3. Determine the winning image
    if conf_180 > conf_0:
        print(f"   -> Image was upside down. Rotated 180°")
        best_image = flipped_image
    else:
        print(f"   -> Image orientation is correct.")
        best_image = bw_image_array

    # --- PHASE 2: Group Rows & Extract (Tuned Mode) ---
    print("   -> Applying horizontal grouping tuning...")    
    final_grouped_results = ocr_model.readtext(
        best_image,
        paragraph=True,
        x_ths=1.5,
        y_ths=0.1
    )
    return best_image,final_grouped_results 
    
def format_extracted_text(raw_ocr_results):
    """
    Strictly handles formatting the raw OCR data into a clean dictionary.
    """
    extracted_data = []
    for line in raw_ocr_results:
        text = line[1]
        confidence = line[2] if len(line) > 2 else 0.90
        
        extracted_data.append({
            "text": text, 
            "confidence": float(confidence)})
    return extracted_data