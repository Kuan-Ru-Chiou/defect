import csv
import os
import subprocess

import chardet
from loguru import logger


def detect_encoding(file_path):
    """Detects the encoding of a file."""
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        return result['encoding']

def extract_defect_info(file_path):
    """Extracts the defect number, class type, and total count from an .lrf file."""
    try:
        encoding = detect_encoding(file_path)
        defects = []
        total_defects = 0
        with open(file_path, encoding=encoding, errors='ignore') as file:
            bw_defect_table = False
            header_found = False
            
            for line in file:
                line = line.strip()
                if "[DefectList]" in line or "DefectDataList" in line:
                    bw_defect_table = True
                    continue
                    
                if bw_defect_table:
                    if line.startswith("DefectDataList"):
                        # Extract total defect count from the line (e.g., "DefectDataList 22")
                        parts = line.split()
                        if len(parts) == 2 and parts[1].isdigit():
                            total_defects = int(parts[1])
                        continue
                        
                    if not header_found:
                        # Skip the header line
                        header_found = True
                        continue
                    elif not line:  # Skip empty lines
                        continue
                        
                    # Split line and filter out empty strings
                    parts = [p for p in line.split(' ') if p]
                    if len(parts) >= 10:  # Ensure line has enough parts
                        defect_num = parts[0]
                        class_type = parts[8]  # ClassType is typically the 9th column (0-based index)
                        if defect_num.isdigit() and class_type.lstrip('-').isdigit():  # Allow for negative numbers
                            defects.append({
                                'number': int(defect_num),
                                'class_type': int(class_type)
                            })
                                
        # If no total_defects was found in DefectDataList, use the count of defects found
        if total_defects == 0:
            total_defects = len(defects)
            
        return defects, total_defects
    except Exception as e:
        logger.error(f"Error reading or processing {file_path}: {e}")
        with open('error.log', 'a') as log_file:
            log_file.write(f"{file_path}: {e}\n")
        return None, 0

def count_images(lot_path):
    """Count the number of image files in the InstantReviewRt directory"""
    try:
        rt_path = os.path.join(os.path.dirname(lot_path), 'InstantReviewRt/Images')
        if os.path.exists(rt_path):
            # Count only .png files that don't have L, L_p, U, or U_p suffixes
            image_files = [f for f in os.listdir(rt_path) 
                         if f.endswith('.png') 
                         and not any(suffix in f for suffix in ['L.png', 'L_p.png', 'U.png', 'U_p.png'])]
            return len(image_files)
        return 0
    except Exception as e:
        logger.error(f"Error counting images in {lot_path}: {e}")
        return 0

def find_lrf_files_and_extract_info(directory):
    """Finds all .lrf files in the directory and its subdirectories, extracts lot IDs,
    finds associated defect numbers, and counts image files."""
    lrf_info = []
    
    # Walk through directory and subdirectories
    for root, _, files in os.walk(directory, followlinks=True):
        for file in files:
            if file.endswith('.lrf'):
                # Check if file has one of the required suffixes
                if not any(suffix in file for suffix in ['_ADD', '_classify', '_classified','_Classified', '_for_Ken']):
                    continue
                    
                file_path = os.path.join(root, file)
                lot_id = os.path.splitext(os.path.basename(file_path))[0]
                logger.info(f"Processing {lot_id}")
                
                defect_info, total_defects = extract_defect_info(file_path)
                image_count = count_images(file_path)
                
                if defect_info:
                    for defect in defect_info:
                        lrf_info.append({
                            'lot_id': lot_id,
                            'defect_number': defect['number'],
                            'image_count': image_count,
                            'total_defects': total_defects,
                            'class_type': defect['class_type']
                        })
                else:
                    # If no valid defect info found, count it as class 0
                    logger.warning(f"No valid defect info found for {lot_id} - treating as Class 0")
                    lrf_info.append({
                        'lot_id': lot_id,
                        'defect_number': 1,  # Assign a dummy defect number
                        'image_count': image_count,
                        'total_defects': 1,  # Set total defects to 1 since we're counting it as one class 0 defect
                        'class_type': 0  # Count as Class 0
                    })

    return lrf_info

def write_to_csv(lrf_info, output_file):
    """Writes the aggregated lot information to a CSV file."""
    try:
        # Aggregate information by lot
        lot_summary = {}
        total_all_defects = 0
        total_class_0 = 0
        total_class_1 = 0
        total_class_2 = 0
        total_class_10 = 0
        total_class_minus1 = 0
        
        for info in lrf_info:
            lot_id = info['lot_id']
            if lot_id not in lot_summary:
                lot_summary[lot_id] = {
                    'total_defects': info['total_defects'],
                    'image_count': info['image_count'],
                    'class_0_count': 0,
                    'class_1_count': 0,
                    'class_2_count': 0,
                    'class_10_count': 0,
                    'class_minus1_count': 0
                }
            # Count defects by class
            if info['class_type'] == 0:
                lot_summary[lot_id]['class_0_count'] += 1
            elif info['class_type'] == 1:
                lot_summary[lot_id]['class_1_count'] += 1
            elif info['class_type'] == 2:
                lot_summary[lot_id]['class_2_count'] += 1
            elif info['class_type'] == 10:
                lot_summary[lot_id]['class_10_count'] += 1
            elif info['class_type'] == -1:
                lot_summary[lot_id]['class_minus1_count'] += 1
            else:
                logger.info(f"Unknown class type: {info['class_type']} in Lot: {lot_id}")
        # Write aggregated information to CSV
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['Lot_ID', 'Total_Defects', 'Class_0_Count', 'Class_1_Count', 'Class_2_Count', 'Class_10_Count', 'Class_minus1_Count', 'Image_Count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for lot_id, summary in lot_summary.items():
                writer.writerow({
                    'Lot_ID': lot_id,
                    'Total_Defects': summary['total_defects'],
                    'Class_0_Count': summary['class_0_count'],
                    'Class_1_Count': summary['class_1_count'],
                    'Class_2_Count': summary['class_2_count'],
                    'Class_minus1_Count': summary['class_minus1_count'],
                    'Class_10_Count': summary['class_10_count'],
                    'Image_Count': summary['image_count']
                })
                total_all_defects += summary['total_defects']
                total_class_0 += summary['class_0_count']
                total_class_1 += summary['class_1_count']
                total_class_2 += summary['class_2_count']
                total_class_10 += summary['class_10_count']
                total_class_minus1 += summary['class_minus1_count']
            
            # Write totals row
            writer.writerow({
                'Lot_ID': 'TOTAL',
                'Total_Defects': total_all_defects,
                'Class_0_Count': total_class_0,
                'Class_1_Count': total_class_1,
                'Class_2_Count': total_class_2,
                'Class_10_Count': total_class_10,
                'Class_minus1_Count': total_class_minus1,
                'Image_Count': 0
            })
            
        # Print totals to console
        print(f"\nTotals across all lots:")
        print(f"Total Defects: {total_all_defects}")
        print(f"Total Class 0: {total_class_0}")
        print(f"Total Class 1: {total_class_1}")
        print(f"Total Class 2: {total_class_2}")
        print(f"Total Class 10: {total_class_10}")
        print(f"Total Class -1: {total_class_minus1}")
        
    except Exception as e:
        logger.error(f"Error writing to CSV {output_file}: {e}")

if __name__ == "__main__":
    dataset_dir = '.'
    lrf_info = find_lrf_files_and_extract_info(dataset_dir)
    output_file = 'lrf_defect_info.csv'
    write_to_csv(lrf_info, output_file)
    print(f"Information saved to {output_file}")
