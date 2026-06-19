# generate_lr.py

import cv2
import os

hr_folder = "dataset/HR"
lr_folder = "dataset/LR"

os.makedirs(lr_folder, exist_ok=True)

for img_name in os.listdir(hr_folder):
    img = cv2.imread(os.path.join(hr_folder, img_name))

    h, w = img.shape[:2]

    lr = cv2.resize(
        img,
        (w//4, h//4),
        interpolation=cv2.INTER_CUBIC
    )

    cv2.imwrite(
        os.path.join(lr_folder, img_name),
        lr
    )

print("Done")