import os
import cv2

current_dir = os.path.dirname(__file__)
image_file_path = os.path.abspath(f"{current_dir}/../himshikhar26/7_digit.png")

image = cv2.imread(image_file_path)
print(image.shape)