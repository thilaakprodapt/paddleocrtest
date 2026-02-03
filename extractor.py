"""
CIOMS Form Text Extractor using PaddleOCR 3.x
Extracts key-value pairs from CIOMS forms, outputting only filled fields.
Handles scanned images, digital documents, and handwritten forms.

Run on GCP VM (Linux):
    1. Upload this file and your image to the VM
    2. pip install paddleocr paddlepaddle
    3. python extractor.py <image_path>
"""

import os
import sys
import re
from typing import Dict, List, Tuple, Optional

from paddleocr import PaddleOCR


def initialize_ocr() -> PaddleOCR:
    """Initialize PaddleOCR with English language support."""
    return PaddleOCR(lang='en')


def extract_text_from_image(ocr: PaddleOCR, image_path: str) -> List:
    """Extract all text from an image using PaddleOCR."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Use predict() method for PaddleOCR 3.x
    result = ocr.predict(image_path)
    return result if result else []


def normalize_ocr_results(ocr_results: List) -> List[dict]:
    """Normalize OCR results to a common format."""
    normalized = []
    
    if not ocr_results:
        return normalized
    
    # PaddleOCR 3.x returns list of result objects
    for result in ocr_results:
        if hasattr(result, 'rec_texts') and hasattr(result, 'dt_polys'):
            # New format with attributes
            texts = result.rec_texts if result.rec_texts else []
            boxes = result.dt_polys if result.dt_polys else []
            scores = result.rec_scores if hasattr(result, 'rec_scores') else []
            
            for i, text in enumerate(texts):
                box = boxes[i] if i < len(boxes) else [[0,0], [0,0], [0,0], [0,0]]
                confidence = scores[i] if i < len(scores) else 0.0
                normalized.append({'text': text, 'box': box, 'confidence': confidence})
        elif isinstance(result, dict):
            # Dict format
            texts = result.get('rec_texts', [])
            boxes = result.get('dt_polys', [])
            scores = result.get('rec_scores', [])
            
            for i, text in enumerate(texts):
                box = boxes[i] if i < len(boxes) else [[0,0], [0,0], [0,0], [0,0]]
                confidence = scores[i] if i < len(scores) else 0.0
                normalized.append({'text': text, 'box': box, 'confidence': confidence})
        elif isinstance(result, (list, tuple)) and len(result) >= 2:
            # Old format: [box, (text, confidence)]
            box = result[0]
            text_info = result[1]
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                text, confidence = text_info[0], text_info[1]
            else:
                text, confidence = str(text_info), 0.0
            normalized.append({'text': text, 'box': box, 'confidence': confidence})
    
    return normalized


def get_text_position(item: dict) -> Tuple[float, float, float, float]:
    """Extract position information from normalized OCR result."""
    box = item.get('box', [[0,0], [0,0], [0,0], [0,0]])
    try:
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    except:
        return (0, 0, 0, 0)


def is_likely_filled(text: str) -> bool:
    """Determine if a text value appears to be filled."""
    if not text or not text.strip():
        return False
    
    empty_indicators = ['-', '--', '---', '____', '....', '…', 
                        'n/a', 'na', 'nil', 'none', 'empty',
                        '[ ]', '( )', '[  ]', '(  )']
    
    normalized = text.strip().lower()
    if normalized in empty_indicators:
        return False
    if re.match(r'^[_.\-\s]+$', text):
        return False
    if not re.search(r'[a-zA-Z0-9]', text):
        return False
    
    return True


def group_text_by_lines(ocr_results: List[dict], y_threshold: float = 15) -> List[List[dict]]:
    """Group OCR results into lines based on vertical position."""
    if not ocr_results:
        return []
    
    sorted_results = sorted(ocr_results, key=lambda x: get_text_position(x)[1])
    lines = []
    current_line = [sorted_results[0]]
    current_y = get_text_position(sorted_results[0])[1]
    
    for result in sorted_results[1:]:
        y_pos = get_text_position(result)[1]
        if abs(y_pos - current_y) <= y_threshold:
            current_line.append(result)
        else:
            current_line.sort(key=lambda x: get_text_position(x)[0])
            lines.append(current_line)
            current_line = [result]
            current_y = y_pos
    
    current_line.sort(key=lambda x: get_text_position(x)[0])
    lines.append(current_line)
    return lines


# CIOMS form field patterns
CIOMS_FIELD_PATTERNS = [
    (r'patient\s*(?:initials?|init)', 'Patient Initials'),
    (r'date\s*of\s*birth|dob|birth\s*date', 'Date of Birth'),
    (r'age\s*(?:at\s*event)?', 'Age'),
    (r'sex|gender', 'Sex'),
    (r'weight', 'Weight'),
    (r'height', 'Height'),
    (r'reaction\s*(?:description)?|adverse\s*(?:event|reaction)', 'Adverse Reaction'),
    (r'(?:reaction\s*)?onset\s*date|date\s*of\s*(?:onset|event)', 'Onset Date'),
    (r'(?:reaction\s*)?end\s*date|date\s*of\s*(?:recovery|resolution)', 'End Date'),
    (r'outcome|resulted?\s*in', 'Outcome'),
    (r'suspect(?:ed)?\s*drug|product\s*name|drug\s*name|medication', 'Suspect Drug'),
    (r'(?:daily\s*)?dose|dosage', 'Dose'),
    (r'route\s*(?:of\s*administration)?', 'Route of Administration'),
    (r'indication|reason\s*for\s*use', 'Indication'),
    (r'therapy\s*dates?|(?:start|begin)\s*date', 'Therapy Start Date'),
    (r'(?:stop|end)\s*date|therapy\s*end', 'Therapy End Date'),
    (r'batch|lot\s*(?:number|no)?', 'Batch/Lot Number'),
    (r'manufacturer', 'Manufacturer'),
    (r'reporter\s*(?:name)?|reported\s*by', 'Reporter Name'),
    (r'report\s*(?:date|received)', 'Report Date'),
    (r'(?:telephone|phone|tel)(?:\s*(?:no|number))?', 'Telephone'),
    (r'case\s*(?:number|no|id)', 'Case Number'),
    (r'(?:country|nation)', 'Country'),
    (r'serious(?:ness)?', 'Seriousness'),
    (r'medical\s*history|relevant\s*history', 'Medical History'),
    (r'concomitant\s*(?:drugs?|medications?|therapy)', 'Concomitant Medications'),
    (r'comments?|remarks?|notes?', 'Comments'),
    (r'action\s*taken', 'Action Taken'),
]


def identify_field_label(text: str) -> Optional[str]:
    """Identify if text matches a known CIOMS field label."""
    normalized = text.lower().strip()
    for pattern, field_name in CIOMS_FIELD_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return field_name
    return None


def extract_key_value_pairs(lines: List[List[dict]]) -> Dict[str, str]:
    """Extract key-value pairs from grouped text lines."""
    extracted_pairs = {}
    current_key = None
    current_value_parts = []
    
    for line in lines:
        line_text = ' '.join([item['text'] for item in line])
        field_name = identify_field_label(line_text)
        
        if field_name:
            if current_key and current_value_parts:
                value = ' '.join(current_value_parts).strip()
                if is_likely_filled(value):
                    extracted_pairs[current_key] = value
            
            current_key = field_name
            for item in line:
                item_text = item['text']
                if not identify_field_label(item_text) and is_likely_filled(item_text):
                    current_value_parts = [item_text]
                    break
            else:
                current_value_parts = []
        
        elif current_key:
            value_text = line_text.strip()
            if value_text and not identify_field_label(value_text):
                current_value_parts.append(value_text)
        
        else:
            for item in line:
                text = item['text']
                if ':' in text:
                    parts = text.split(':', 1)
                    if len(parts) == 2:
                        key, value = parts[0].strip(), parts[1].strip()
                        if is_likely_filled(value) and len(key) > 1:
                            std_field = identify_field_label(key)
                            extracted_pairs[std_field if std_field else key] = value
    
    if current_key and current_value_parts:
        value = ' '.join(current_value_parts).strip()
        if is_likely_filled(value):
            extracted_pairs[current_key] = value
    
    return extracted_pairs


def extract_structured_content(lines: List[List[dict]]) -> Dict[str, str]:
    """Extract content by analyzing spatial relationships."""
    extracted_pairs = {}
    
    for line in lines:
        if len(line) < 1:
            continue
        
        items = [(item['text'], get_text_position(item)) for item in line]
        
        for i, (text, pos) in enumerate(items):
            field_name = identify_field_label(text)
            
            if field_name:
                x_max = pos[2]
                for j, (other_text, other_pos) in enumerate(items):
                    if j > i and other_pos[0] > x_max:
                        if is_likely_filled(other_text) and not identify_field_label(other_text):
                            extracted_pairs[field_name] = other_text
                            break
            
            if ':' in text:
                parts = text.split(':', 1)
                if len(parts) == 2:
                    key, value = parts[0].strip(), parts[1].strip()
                    if key and is_likely_filled(value):
                        std_field = identify_field_label(key)
                        extracted_pairs[std_field if std_field else key] = value
    
    return extracted_pairs


def post_process_extracted_data(data: Dict[str, str]) -> Dict[str, str]:
    """Clean and validate extracted data."""
    cleaned = {}
    for key, value in data.items():
        if not is_likely_filled(value):
            continue
        cleaned_value = re.sub(r'\s+', ' ', value.strip())
        if len(cleaned_value) > 0:
            cleaned[key] = cleaned_value
    return cleaned


def extract_cioms_data(image_path: str, verbose: bool = False) -> Dict[str, str]:
    """Main function to extract key-value pairs from a CIOMS form image."""
    ocr = initialize_ocr()
    
    if verbose:
        print(f"Extracting text from: {image_path}")
    
    raw_results = extract_text_from_image(ocr, image_path)
    ocr_results = normalize_ocr_results(raw_results)
    
    if verbose:
        print(f"Found {len(ocr_results)} text elements")
    
    if not ocr_results:
        print("No text found in image")
        return {}
    
    lines = group_text_by_lines(ocr_results)
    
    if verbose:
        print(f"Grouped into {len(lines)} lines")
        print("\n--- Raw OCR Text ---")
        for line in lines:
            print(' '.join([item['text'] for item in line]))
        print("--- End Raw OCR ---\n")
    
    extracted_data = {}
    extracted_data.update(extract_key_value_pairs(lines))
    
    spatial_based = extract_structured_content(lines)
    for key, value in spatial_based.items():
        if key not in extracted_data:
            extracted_data[key] = value
    
    return post_process_extracted_data(extracted_data)


def format_output(data: Dict[str, str]) -> str:
    """Format extracted data for display."""
    if not data:
        return "No filled fields found in the form."
    
    output_lines = ["=" * 60, "CIOMS FORM - EXTRACTED DATA (FILLED FIELDS ONLY)", "=" * 60]
    
    categories = {
        'Patient Information': ['Patient Initials', 'Date of Birth', 'Age', 'Sex', 'Weight', 'Height'],
        'Adverse Event': ['Adverse Reaction', 'Onset Date', 'End Date', 'Outcome', 'Seriousness'],
        'Drug Information': ['Suspect Drug', 'Dose', 'Route of Administration', 'Indication', 
                            'Therapy Start Date', 'Therapy End Date', 'Batch/Lot Number', 'Manufacturer'],
        'Reporter Information': ['Reporter Name', 'Report Date', 'Telephone'],
        'Case Details': ['Case Number', 'Country', 'Action Taken'],
        'Additional Information': ['Medical History', 'Concomitant Medications', 'Comments']
    }
    
    categorized_keys = set()
    
    for category, fields in categories.items():
        category_data = {k: v for k, v in data.items() if k in fields}
        if category_data:
            output_lines.append(f"\n{category}")
            output_lines.append("-" * len(category))
            for key, value in category_data.items():
                output_lines.append(f"  {key}: {value}")
                categorized_keys.add(key)
    
    uncategorized = {k: v for k, v in data.items() if k not in categorized_keys}
    if uncategorized:
        output_lines.append("\nOther Fields")
        output_lines.append("-" * 12)
        for key, value in uncategorized.items():
            output_lines.append(f"  {key}: {value}")
    
    output_lines.append("\n" + "=" * 60)
    return '\n'.join(output_lines)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for default image
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_files = ['Filled_Cioms_form-1.jpg', 'cioms_form.jpg', 'cioms_form.png', 'cioms.jpg', 'form.jpg']
        
        image_path = None
        for filename in possible_files:
            full_path = os.path.join(current_dir, filename)
            if os.path.exists(full_path):
                image_path = full_path
                break
        
        if not image_path:
            print("Usage: python extractor.py <image_path>")
            sys.exit(1)
    
    print(f"Processing: {image_path}")
    print("Please wait, extracting text...\n")
    
    try:
        extracted_data = extract_cioms_data(image_path, verbose=True)
        print(format_output(extracted_data))
        return extracted_data
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()