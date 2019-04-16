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

def distance_converter(image_dist, real_dist, *args):
    result = []
    conversion_rate = real_dist/float(image_dist)
    for arg in args:
        result.append(arg*conversion_rate)
    return tuple(result)

def euclidan_dist(*args):
    # Computes euclidan distance of pairs of coordinates 
    result = 0
    if len(args) % 2 != 0:
        return 0        
    for i in range(0, len(args), 2):
        result += (args[i] - args[i+1])**2
    return result ** 0.5

def center_coords(x_start, length):
    return int(x_start + length/2)
