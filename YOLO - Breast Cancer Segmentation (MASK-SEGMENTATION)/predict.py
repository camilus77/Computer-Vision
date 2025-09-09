from ultralytics import YOLO

import cv2


model_path = 'runs/segment/train2/weights/last.pt'

image_path = 'C:/Users/ben4s/Downloads/Breast Cancer Segmentation using YOLO/Dataset/malignant/images/train/malignant (97).png'

img = cv2.imread(image_path)
print(img)
H, W, _ = img.shape
model = YOLO(model_path)

results = model(img)
for result in results:
    for j, mask in enumerate(result.masks.data):

        mask = mask.numpy() * 255

        mask = cv2.resize(mask, (W, H))

        cv2.imwrite('./output.png', mask)

