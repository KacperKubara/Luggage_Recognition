# Helper function for the project
import cv2
import os

def load_image(image_path = "./test_data/luggage0.jpeg"):
    return cv2.imread(image_path)

def load_yolo(relative_path = "./yolo-detector"):

    # derive the relative_paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([relative_path, "yolov3.weights"])
    configPath = os.path.sep.join([relative_path, "yolov3.cfg"])

    print("[INFO] Loading YOLO from disk...")
    return cv2.dnn.readNetFromDarknet(configPath, weightsPath)