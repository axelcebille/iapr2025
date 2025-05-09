import os
import json
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import matplotlib.pyplot as plt



PROJECT_DIR = os.path.dirname(__file__)  # gets the directory of b_boxes.py
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')

TRAIN_JSON = os.path.join(DATA_DIR, 'train.json')
VAL_JSON = os.path.join(DATA_DIR, 'valid.json')

TRAIN_IMG_DIR = os.path.join(IMAGES_DIR, 'train')
VAL_IMG_DIR = os.path.join(IMAGES_DIR, 'valid')

NUM_CLASSES = 1 + 13  # Replace N with number of chocolate classes


class ChocolateCocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        with open(ann_file) as f:
            data = json.load(f)
        self.images = data['images']
        self.annotations = data['annotations']
        self.categories = data['categories']
        
        # Create image_id → annotations mapping
        self.image_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            self.image_id_to_anns.setdefault(img_id, []).append(ann)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info['id']
        img_path = os.path.join(self.img_dir, image_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        img = T.ToTensor()(img)

        boxes = []
        labels = []

        for ann in self.image_id_to_anns.get(image_id, []):
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])  # Must start from 1

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
        }

        return img, target

    def __len__(self):
        return len(self.images)

def get_data_loaders(batch_size=4):
    train_dataset = ChocolateCocoDataset(TRAIN_IMG_DIR, TRAIN_JSON)
    val_dataset = ChocolateCocoDataset(VAL_IMG_DIR, VAL_JSON)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, val_loader

def get_model(num_classes):
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # return model
    # Load MobileNetV2 backbone (features only)
    backbone = mobilenet_v2(weights="DEFAULT").features
    backbone.out_channels = 1280  # last feature map size

    # Wrap with FPN (optional but improves performance)
    backbone = BackboneWithFPN(backbone, returned_layers=[-1], in_channels_list=[1280], out_channels=256)

    # Create Faster R-CNN model
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

def train(model, dataloader, optimizer, device):
    model.train()
    for imgs, targets in dataloader:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print(f"Loss: {losses.item():.4f}")



def test_model(model, dataloader, device, threshold=0.5):
    import random
    from PIL import ImageDraw

    model.eval()
    imgs, _ = next(iter(dataloader))
    img = imgs[0]
    with torch.no_grad():
        pred = model([img.to(device)])[0]

    img = img.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
        if score > threshold:
            draw.rectangle(box.tolist(), outline='green', width=2)
            draw.text((box[0], box[1]), f"{label.item()} ({score:.2f})", fill='green')

    img.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_data_loaders()
    model = get_model(NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        print(f"Epoch {epoch + 1}")
        train(model, train_loader, optimizer, device)

    torch.save(model.state_dict(), 'fasterrcnn_chocolates.pth')
    print("Model saved.")

    test_model(model, val_loader, device)
    print("Tested model on validation set.")




if __name__ == "__main__":
    main()


