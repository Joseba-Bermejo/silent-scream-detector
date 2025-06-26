import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

# Dummy scaler and feature order for the example
scaler = StandardScaler()
def preprocess_image(img):
    resized = cv2.resize(img, (128, 128))
    flat = resized.flatten().reshape(1, -1)
    scaled = scaler.fit_transform(flat) 
    return scaled