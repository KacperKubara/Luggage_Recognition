# Helper function for the project
import cv2
import os

def load_image(image_name = "./test_data"):
    return cv2.imread(os.path.join("./test_data", image_name))

def load_yolo(relative_path = "./yolo-coco/"):

    # derive the relative_paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([relative_path, "yolov3.weights"])
    configPath = os.path.sep.join([relative_path, "yolov3.cfg"])

    print("[INFO] loading YOLO from disk...")
    return cv2.dnn.readNetFromDarknet(configPath, weightsPath)