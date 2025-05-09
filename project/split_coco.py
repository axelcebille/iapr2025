# import json, random, os

# INPUT_JSON = "data/_annotations.coco.json"
# TRAIN_JSON = "data/train.json"
# VAL_JSON = "data/valid.json"
# VAL_RATIO = 0.2  # 20% for validation

# with open(INPUT_JSON, "r") as f:
#     coco = json.load(f)

# images = coco["images"]
# annotations = coco["annotations"]
# categories = coco["categories"]

# random.shuffle(images)

# val_size = int(len(images) * VAL_RATIO)
# val_images = images[:val_size]
# train_images = images[val_size:]

# val_ids = set(img["id"] for img in val_images)
# train_ids = set(img["id"] for img in train_images)

# def filter_anns(image_ids):
#     return [ann for ann in annotations if ann["image_id"] in image_ids]

# train_data = {
#     "images": train_images,
#     "annotations": filter_anns(train_ids),
#     "categories": categories,
# }
# val_data = {
#     "images": val_images,
#     "annotations": filter_anns(val_ids),
#     "categories": categories,
# }

# with open(TRAIN_JSON, "w") as f:
#     json.dump(train_data, f)

# with open(VAL_JSON, "w") as f:
#     json.dump(val_data, f)

# print(f"Saved {len(train_images)} train images and {len(val_images)} val images.")

import json, os, shutil

ANNOTATION_DIR = "data"
IMAGE_SOURCE_DIR = "images"
TRAIN_IMG_DIR = os.path.join(IMAGE_SOURCE_DIR, "train")
VAL_IMG_DIR = os.path.join(IMAGE_SOURCE_DIR, "valid")

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)

def move_images(json_path, dest_dir):
    with open(json_path, "r") as f:
        coco = json.load(f)
    for img in coco["images"]:
        filename = img["file_name"]
        src_path = os.path.join(IMAGE_SOURCE_DIR, filename)
        dst_path = os.path.join(dest_dir, filename)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Warning: {src_path} not found.")

move_images(os.path.join(ANNOTATION_DIR, "train.json"), TRAIN_IMG_DIR)
move_images(os.path.join(ANNOTATION_DIR, "valid.json"), VAL_IMG_DIR)



