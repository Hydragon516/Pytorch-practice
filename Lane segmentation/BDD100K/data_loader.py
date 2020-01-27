import pickle
import numpy as np
import cv2
import os, glob

img_dir = sorted(glob.glob('./bdd100k_images/bdd100k/images/100k/val/*'))
mask_dir = sorted(glob.glob('./bdd100k_drivable_maps/bdd100k/drivable_maps/color_labels/val/*'))

from tqdm import tqdm_notebook

total1 = tqdm_notebook(img_dir)
total2 = tqdm_notebook(mask_dir)

height = 80
width = 160

image = []
mask = []

for img_path in total1:    
        
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    img = np.transpose(img, (2, 0, 1)) / 255
    image.append(img)

for mask_path in total2:    
        
    img = cv2.imread(mask_path, 1)
    lower = np.array([0, 0, 254], dtype="uint8")
    upper = np.array([0, 0, 255], dtype="uint8")

    masks = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=masks)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(output, (width, height), interpolation=cv2.INTER_CUBIC)
    img = img.reshape(1, height, width) / 255
    mask.append(img)
        
image = np.array(image)
mask = np.array(mask)

with open('./pickle/image_train.p', 'wb') as f:
    pickle.dump(image, f, protocol=4)

with open('./pickle/mask_labels.p', 'wb') as f:
    pickle.dump(mask, f, protocol=4)