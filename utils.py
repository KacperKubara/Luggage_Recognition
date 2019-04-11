# Helper function for the project
import cv2
import os
def load_image(image_name):
    return cv2.imread(os.path.join("./test_data", image_name))
