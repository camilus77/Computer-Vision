import cv2

import os
import shutil
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt


img_path = 'C:\\Users\\ben4s\\Downloads\\Dataset_BUSI_with_GT\\malignant\\malignant (93)\\images' 
image = cv2.imread(img_path)

if image is not None:
    height, width = image.shape[:2]
    print(f"Shape: Width: {width}, Height: {height}")
else:
    print(f"Image not found: {img_path}")


def make_pandas_for_size(base_dir: str):
    image_sizes = []
    file_list = [f for f in os.listdir(base_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for file_name in tqdm(file_list, desc="Image not found"):
        img_path = os.path.join(base_dir, file_name)
        image = cv2.imread(img_path)
        if image is not None:
            h, w = image.shape[:2]
            image_sizes.append({'image_path': img_path, 'width': w, 'height': h})
        else:
            print("Cannot read image: ", img_path)

    return pd.DataFrame(image_sizes)

base_dir = 'C:\\Users\\ben4s\\Downloads\\Dataset_BUSI_with_GT\\malignant\\images'
df = make_pandas_for_size(base_dir)
print(df)



min_width = df['width'].min()
min_height = df['height'].min()
print("Image with smallest dimension: ")
print(f"smallest height:{min_height} smallest width:{min_width}")


def img_mask_to_yoloformat(
    image_dir: str,
    mask_dir: str,
    output_dir: str = 'labels',
    class_id: int = 0,
    image_ext: str = '.png'
) -> None:
    """
    It creates segmentation labels in YOLO format from ultrasound images and masks..

    Args:
        image_dir (str): Orijinal görüntülerin bulunduğu dizin.
        mask_dir (str): Maske dosyalarının bulunduğu dizin.
        output_dir (str): Çıktı etiket dosyalarının kaydedileceği dizin.
        class_id (int): Segmentasyon sınıf ID'si (varsayılan: 0).
        image_ext (str): Görüntü dosyalarının uzantısı (varsayılan: '.png').
    """
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Could not create output directory: {e}")
        return
    

    # List mask files
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(image_ext)]
    if not mask_files:
        print(f"{mask_dir} Directory {image_ext} file with extension not found.")
        return

    # İlerleme çubuğu ile maskeleri işle
    for filename in tqdm(mask_files, desc="Processed Masks"):
        try:
            # Render masks with progress bar
            mask_path = os.path.join(mask_dir, filename)
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(output_dir, filename.replace(image_ext, '.txt'))

            # Read image and mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(image_path)

            if mask is None or image is None:
                print(f"Uyarı: {filename} Could not read image or mask for: ")
                continue

            # Görüntü boyutlarını al
            h, w = image.shape[:2]
            if h == 0 or w == 0:
                print(f"Uyarı: {filename} Invalid image dimensions for: ")
                continue

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Write tag file
            with open(label_path, 'w') as f:
                for contour in contours:
                    if len(contour) < 3:
                        continue  # not a polygon

                    # Normalize coordinates
                    normalized_points = []
                    for point in contour.squeeze():
                        x = point[0] / w
                        y = point[1] / h
                        # Make sure it's between 0-1
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            normalized_points.append(f"{x:.6f} {y:.6f}")
                        else:
                            print(f"WARNING: {filename} Invalid coordinate: ({x}, {y})")

                    if normalized_points:
                        polygon_str = ' '.join(normalized_points)
                        f.write(f"{class_id} {polygon_str}\n")

        except Exception as e:
            print(f"Hata: {filename} Problem processing file: {e}")
            continue

image_dir = 'C:\\Users\\ben4s\\Downloads\\Dataset_BUSI_with_GT\\malignant\\images'
mask_dir = 'C:\\Users\\ben4s\\Downloads\\Dataset_BUSI_with_GT\\malignant\\masks'

img_mask_to_yoloformat(
    image_dir=image_dir,
    mask_dir=mask_dir,
    output_dir='C:\\Users\\ben4s\\Downloads\\Dataset_BUSI_with_GT\\malignant\\labels',
    class_id=0,
    image_ext='.png'
)


txt_path = 'C:\\Users\\ben4s\\Downloads\\Dataset_BUSI_with_GT\\malignant\\labels\\malignant (4).txt'

with open(txt_path, 'r', encoding='utf-8') as f:
    content = f.read()
    print(content)