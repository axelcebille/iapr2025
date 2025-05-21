import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from glob import glob
import numpy as np
from sklearn.cluster import KMeans


def show_bounding_boxes(image_folder, csv1_path, csv2_path):
    # Normalize all paths to absolute, relative to current working directory (which is src/)
    image_folder = os.path.abspath(image_folder)
    csv1_path = os.path.abspath(csv1_path)
    csv2_path = os.path.abspath(csv2_path)

    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    def get_base_filename(filename):
        return "_".join(filename.split("_")[:2])

    df1['base_filename'] = df1['filename'].apply(get_base_filename)
    df2['base_filename'] = df2['filename'].apply(get_base_filename)

    image_filenames = set(os.listdir(image_folder))
    df1 = df1[df1['filename'].isin(image_filenames)]

    grouped1 = df1.groupby('filename')
    grouped2 = df2.groupby('base_filename')

    num_images = len(grouped1)
    cols = 3
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)[0]
    axes = axes.flatten()

    for idx, (filename, group1) in enumerate(grouped1):
        img_path = os.path.join(image_folder, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax = axes[idx]
        ax.imshow(image)
        ax.set_title(filename, fontsize=8)
        ax.axis('off')

        for _, row in group1.iterrows():
            x, y = row['xmin'], row['ymin']
            w, h = row['xmax'] - row['xmin'], row['ymax'] - row['ymin']
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)

        base_filename = get_base_filename(filename)
        if base_filename in grouped2.groups:
            group2 = grouped2.get_group(base_filename)
            for _, row in group2.iterrows():
                x, y = row['xmin'], row['ymin']
                w, h = row['xmax'] - row['xmin'], row['ymax'] - row['ymin']
                rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def kmeans_comparisons(function, input_folder, max_k=6):
    image_paths = glob(os.path.join(input_folder, "*.jpg"))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}")
            continue

        fig, axs = plt.subplots(1, max_k, figsize=(5 * max_k, 5))
        axs = np.array(axs).reshape(-1)  # Ensure axs is always iterable

        fig.suptitle(f"K-means Clustering: {os.path.basename(img_path)}", fontsize=18)

        for i in range(max_k):
            k = i + 1
            clustered_img = function(img, k)[0]
            rgb_img = cv2.cvtColor(clustered_img, cv2.COLOR_BGR2RGB)
            axs[i].imshow(rgb_img)
            axs[i].set_title(f'k = {k}', fontsize=14)
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()

def draw_contours(image, contours, labels=None):
    boxed = image.copy()
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 6)
        if labels and i < len(labels):
            cv2.putText(boxed, labels[i], (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        2.0, (0, 0, 255), 4, lineType=cv2.LINE_AA)
    return boxed

def display_results(orig, clustered, contours, labels, choc_contours, choc_labels):
    clustered_boxes = draw_contours(clustered, contours)
    original_boxes = draw_contours(orig.copy(), contours)
    original_choc_boxes = draw_contours(orig.copy(), choc_contours)
    labeled_img = draw_contours(orig.copy(), choc_contours, choc_labels)

    fig, axs = plt.subplots(1, 4, figsize=(20, 8))  # Slightly larger figure
    axs[0].imshow(cv2.cvtColor(clustered_boxes, cv2.COLOR_BGR2RGB))
    axs[0].set_title('K-means with BBoxes', fontsize=24)
    axs[1].imshow(cv2.cvtColor(original_boxes, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Original with BBoxes', fontsize=24)
    axs[2].imshow(cv2.cvtColor(original_choc_boxes, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Original with Choc', fontsize=24)
    axs[3].imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    axs[3].set_title('final Classification', fontsize=24)

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    
def visualize_all_mask_cleaning(masks):
    def clean_mask(mask):
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 55))

        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        
        # Optional: strengthen structure further
        dilated = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)
        return dilated
    cleaned_masks = [clean_mask(mask) for mask in masks]
    num_masks = len(masks)

    plt.figure(figsize=(4 * num_masks, 8))

    for i in range(num_masks):
        plt.subplot(2, num_masks, i + 1)
        plt.imshow(masks[i], cmap='gray')
        plt.title(f'Original Mask (Class {i})')
        plt.axis('off')

        plt.subplot(2, num_masks, i + 1 + num_masks)
        plt.imshow(cleaned_masks[i], cmap='gray')
        plt.title(f'Cleaned Mask (Class {i})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

