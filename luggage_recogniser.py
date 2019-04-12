import cv2
import utils
import time 
import numpy as np

class Luggage_Recogniser:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = utils.load_image(image_path)
        print("[INFO] Image loaded")

        self.labels = open("./yolo-detector/coco.names").read().strip().split("\n")
        self.luggage_labels=["backpack", "handbag", "suitcase"]
        print("[INFO] Labels loaded")
        
        self.net = utils.load_yolo()
        print("[INFO] Yolo loaded")

    def is_attended(self):
        #Algorithm here
        pass

# Return the detected object's labels with their coordinates
    def detect_objects(self):
        # Get the output layer names
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(self.image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(self.ln)
        end = time.time()     
        print("Process took {:.6f} seconds".format(end-start))   

    def show_picture(self):
        self.image.imshow()
        cv2.waitKey(1000)