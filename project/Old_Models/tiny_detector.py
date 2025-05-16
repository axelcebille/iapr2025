import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# ---------------------- MODEL ----------------------

class TinyDetector(nn.Module):
    def __init__(self, num_classes=13, grid_size=14):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.output_channels = 5 + num_classes  # [obj, x, y, w, h, ...classes]

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Conv2d(128, self.output_channels, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x.permute(0, 2, 3, 1)  # [B, S, S, C]


# ---------------------- DATASET ----------------------

class COCODetectionDataset(Dataset):
    def __init__(self, img_dir, ann_path, S=14, num_classes=13, transform=None):
        with open(ann_path) as f:
            self.coco = json.load(f)

        self.img_dir = img_dir
        self.S = S
        self.num_classes = num_classes
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.coco['images']}
        self.annotations = self._build_index()

    def _build_index(self):
        image_to_anns = {}
        for ann in self.coco['annotations']:
            image_id = ann['image_id']
            if image_id not in image_to_anns:
                image_to_anns[image_id] = []
            image_to_anns[image_id].append(ann)
        return image_to_anns

    def __len__(self):
        return len(self.coco['images'])

    def __getitem__(self, idx):
        image_info = self.coco['images'][idx]
        img_path = os.path.join(self.img_dir, image_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        H, W = 224, 224

        label = torch.zeros((self.S, self.S, 5 + self.num_classes))
        anns = self.annotations.get(image_info['id'], [])
        # STOP
        for ann in anns:
            x, y, w, h = ann['bbox']
            class_idx = ann['category_id'] - 1  # Subtract 1 to make category_id range from 0 to 12
            if class_idx >= self.num_classes:
                raise ValueError(f"Invalid category_id {class_idx} in annotation for image {image_info['file_name']}")

            # Normalization and grid assignment as before
            cx = x + w / 2
            cy = y + h / 2
            cx /= W
            cy /= H
            w /= W
            h /= H
            grid_x = int(cx * self.S)
            grid_y = int(cy * self.S)

            if grid_x >= self.S: grid_x = self.S - 1
            if grid_y >= self.S: grid_y = self.S - 1

            label[grid_y, grid_x, 0] = 1
            label[grid_y, grid_x, 1:5] = torch.tensor([cx, cy, w, h])
            label[grid_y, grid_x, 5 + class_idx] = 1  # Correct index for the chocolate class


        # START
        # for ann in anns:
        #     x, y, w, h = ann['bbox']
        #     class_idx = ann['category_id']
        #     cx = x + w / 2
        #     cy = y + h / 2
        #     cx /= W
        #     cy /= H
        #     w /= W
        #     h /= H
        #     grid_x = int(cx * self.S)
        #     grid_y = int(cy * self.S)

        #     if grid_x >= self.S: grid_x = self.S - 1
        #     if grid_y >= self.S: grid_y = self.S - 1

        #     label[grid_y, grid_x, 0] = 1
        #     label[grid_y, grid_x, 1:5] = torch.tensor([cx, cy, w, h])
        #     label[grid_y, grid_x, 5 + class_idx] = 1

        return img, label


# ---------------------- LOSS ----------------------

class YoloLoss(nn.Module):
    def __init__(self, S=14, C=13, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, preds, targets):
        obj_mask = targets[..., 0] == 1
        noobj_mask = targets[..., 0] == 0

        loss_obj = F.binary_cross_entropy_with_logits(preds[..., 0][obj_mask], targets[..., 0][obj_mask])
        loss_noobj = F.binary_cross_entropy_with_logits(preds[..., 0][noobj_mask], targets[..., 0][noobj_mask])

        coord_loss = F.mse_loss(torch.sigmoid(preds[..., 1:5][obj_mask]), targets[..., 1:5][obj_mask])

        class_loss = F.cross_entropy(
            preds[..., 5:][obj_mask].reshape(-1, self.C),
            targets[..., 5:][obj_mask].argmax(-1).reshape(-1)
        )

        return self.lambda_coord * coord_loss + loss_obj + self.lambda_noobj * loss_noobj + class_loss


# ---------------------- TRAINING ----------------------

def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = YoloLoss(S=14, C=model.num_classes)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        total_loss = 0
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        evaluate(model, val_loader, device)


# ---------------------- EVALUATION ----------------------

def evaluate(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        total = 0
        detected = 0
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            preds = model(imgs)
            pred_obj = torch.sigmoid(preds[..., 0])
            total += targets[..., 0].sum().item()
            detected += (pred_obj > 0.5).sum().item()
        print(f"Detected {detected}/{int(total)} objects (≈{(detected/total)*100:.1f}%)")


def test_model(model, dataloader, device, threshold=0.5):
    from PIL import ImageDraw, ImageFont
    import matplotlib.pyplot as plt

    model.eval()
    model.to(device)

    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        with torch.no_grad():
            preds = model(imgs)

        for i in range(imgs.size(0)):
            # img = imgs[i].cpu().permute(1, 2, 0).numpy() * 255
            # img = Image.fromarray(img.astype(np.uint8))
            img = imgs[i].cpu().permute(1, 2, 0).numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img)
            #---
            draw = ImageDraw.Draw(img)

            pred = preds[i].cpu()  # [S, S, C]
            S = pred.shape[0]
            for y in range(S):
                for x in range(S):
                    cell = pred[y, x]
                    obj_score = torch.sigmoid(cell[0])
                    if obj_score > threshold:
                        bx = torch.sigmoid(cell[1]) + x
                        by = torch.sigmoid(cell[2]) + y
                        # bw = torch.sigmoid(cell[3])
                        # bh = torch.sigmoid(cell[4])
                        bw = cell[3]
                        bh = cell[4]
                        bx /= S
                        by /= S
                        bw /= S
                        bh /= S

                        img_w, img_h = img.size
                        x1 = int((bx - bw / 2) * img_w)
                        y1 = int((by - bh / 2) * img_h)
                        x2 = int((bx + bw / 2) * img_w)
                        y2 = int((by + bh / 2) * img_h)

                        class_probs = cell[5:]
                        class_idx = class_probs.argmax().item()
                        score = obj_score.item()

                        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                        draw.text((x1, y1), f"{class_idx} ({score:.2f})", fill="green")

            img.show()

# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    DATA_DIR = os.path.dirname(__file__)
    TRAIN_JSON = os.path.join(DATA_DIR, 'data/train.json')
    VAL_JSON = os.path.join(DATA_DIR, 'data/valid.json')
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'images/train')
    VAL_IMG_DIR = os.path.join(DATA_DIR, 'images/valid')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = COCODetectionDataset(TRAIN_IMG_DIR, TRAIN_JSON, num_classes=13)
    val_ds = COCODetectionDataset(VAL_IMG_DIR, VAL_JSON, num_classes=13)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4)

    model = TinyDetector(num_classes=13)
    # model.load_state_dict(torch.load('tiny_detector.pth'))  # Load pre-trained weights ifavailable
    train_model(model, train_loader, val_loader, device, epochs=10)

    torch.save(model.state_dict(), 'tiny_detector.pth')

    test_model(model, val_loader, device)
    print("Tested model on validation set.")