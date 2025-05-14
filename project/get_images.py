import json, os, shutil

ANNOTATION_DIR = os.path.join(os.path.dirname(__file__), "data")
IMAGE_SOURCE_DIR = os.path.join(os.path.dirname(__file__), "images/train_robo_1")
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
move_images(os.path.join(ANNOTATION_DIR, "valid.json"),VAL_IMG_DIR)