import json
import argparse
from pathlib import Path
from tqdm import tqdm


def remap_annotations(input_file: Path, output_file: Path):
    """
    Loads an STF COCO JSON, merges 'Car' (1) and 'Truck' (3) into a new 'TYPE_VEHICLE' (1) class and remaps other classes to match Waymo format.
    """
    print(f"Loading original file: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Old STF Category IDs:
    # 1: Car
    # 2: Pedestrian
    # 3: Truck
    # 4: Cyclist
    
    # New Category IDs:
    # 1: TYPE_VEHICLE (merges 1 and 3)
    # 2: TYPE_PEDESTRIAN (from 2)
    # 3: TYPE_CYCLIST (from 4)
    
    # A map to convert old IDs to new IDs
    # {old_id: new_id}
    category_id_map = {
        1: 1,
        3: 1,
        2: 2,
        4: 3
    }

    new_categories = [
        {"id": 1, "name": "TYPE_VEHICLE"},
        {"id": 2, "name": "TYPE_PEDESTRIAN"},
        {"id": 3, "name": "TYPE_CYCLIST"}
    ]
    
    new_annotations = []
    print("Remapping annotations...")
    
    for ann in tqdm(data['annotations']):
        old_cat_id = ann['category_id']
        
        if old_cat_id in category_id_map:
            new_cat_id = category_id_map[old_cat_id]
            ann['category_id'] = new_cat_id
            new_annotations.append(ann)

    new_data = {
        "images": data['images'],
        "annotations": new_annotations,
        "categories": new_categories
    }
    
    print(f"Saving remapped file to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)
        
    print(f"Successfully created new annotation file with {len(new_annotations)} annotations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remap STF COCO annotations to 3 classes with Waymo naming.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to the original STF COCO JSON file.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path to save the new, remapped COCO JSON file.")
    
    args = parser.parse_args()
    
    remap_annotations(args.input, args.output)
    
