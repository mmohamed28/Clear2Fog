import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np


def get_image_info(image_path: Path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size

        return {"file_name": image_path.name, "height": height, "width": width}

    except Exception as e:
        print(f"Error opening image {image_path}: {e}")

        return None

def kitti_to_coco(image_dir: Path, label_dir: Path, output_file: Path):
    categories = [
        {"id": 1, "name": "TYPE_VEHICLE"},
        {"id": 2, "name": "TYPE_PEDESTRIAN"},
        {"id": 3, "name": "TYPE_CYCLIST"}
    ]
    
    
    kitti_class_to_coco_id = {
        'car': 1,
        'van': 1, 
        'pedestrian': 2,
        'cyclist': 3
    }
    
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    image_id_counter = 1
    annotation_id_counter = 1

    label_files = sorted(list(label_dir.glob("*.txt")))
    if not label_files:
        print(f"Error: No .txt label files found in {label_dir}")

        return

    print(f"Found {len(label_files)} label files. Starting conversion...")

    for label_path in tqdm(label_files):
        image_name = label_path.stem + ".png"
        image_path = image_dir / image_name
        
        if not image_path.exists():
            print(f"Warning: No matching image for label {label_path.name}. Skipping.")

            continue

        image_info = get_image_info(image_path)
        if image_info is None:
            continue
            
        image_info["id"] = image_id_counter
        coco_output["images"].append(image_info)
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                    
                class_name = parts[0].lower()
                
                if class_name in kitti_class_to_coco_id:
                    coco_cat_id = kitti_class_to_coco_id[class_name]
                    
                    try:
                        bbox_kitti = [float(p) for p in parts[4:8]]
                        
                        x1, y1, x2, y2 = bbox_kitti
                        w = x2 - x1
                        h = y2 - y1
                        
                        if w <= 0 or h <= 0:
                            print(f"Warning: Invalid bbox [w={w}, h={h}] in {label_path.name}. Skipping.")
                            continue
                            
                        bbox_coco = [x1, y1, w, h]
                        
                        annotation = {
                            "id": annotation_id_counter,
                            "image_id": image_id_counter,
                            "category_id": coco_cat_id,
                            "bbox": bbox_coco,
                            "area": w * h,
                            "iscrowd": 0,
                            "segmentation": []
                        }
                        
                        coco_output["annotations"].append(annotation)
                        annotation_id_counter += 1
                        
                    except Exception as e:
                        print(f"Warning: Error parsing line in {label_path.name}: {line}")
                        print(f"  Error: {e}")

        image_id_counter += 1

    print(f"\nConversion complete.")
    print(f"Total Images: {len(coco_output['images'])}")
    print(f"Total Annotations: {len(coco_output['annotations'])}")
    
    with open(output_file, 'w') as f:
        json.dump(coco_output, f, indent=4)
        
    print(f"Successfully saved COCO JSON to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert KITTI format dataset to COCO format.")
    
    parser.add_argument("--image_dir", type=Path, required=True, help="Path to the directory containing the image files.")
    parser.add_argument("--label_dir", type=Path, required=True, help="Path to the directory containing the KITTI .txt label files.")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to save the output COCO .json file.")

    args = parser.parse_args()

    kitti_to_coco(args.image_dir, args.label_dir, args.output_file)
