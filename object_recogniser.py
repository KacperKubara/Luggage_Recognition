import time 

import cv2
import numpy as np

import utils

class ObjectRecogniser:
    def __init__(self, image_path, confidence, threshold):
        self.image_path = image_path
        self.image = utils.load_image(image_path)
        print("[INFO] Image loaded")

        self.confidence = confidence
        self.threshold = threshold
        print("[INFO] Confidence and threshold values set")

        self.labels = open("./yolo-detector/coco.names").read().strip().split("\n")
        self.luggage_labels=["backpack", "handbag", "suitcase"]
        print("[INFO] Labels loaded")
        
        self.net = utils.load_yolo()
        print("[INFO] Yolo loaded")
        
# Return the detected object's labels with their coordinates
    def detect_objects(self):
        self.getLayerNames()
        self.imagePreprocessing()
        self.predict()
        self.filterFigures()
        self.non_overlapping_figures()

        return self.classIDs, self.confidences, self.boxes

    def getLayerNames(self):
        # Get the output layer names
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def imagePreprocessing(self):
        # Preprocess image
        (self.H, self.W) = self.image.shape[:2]
        self.blob = cv2.dnn.blobFromImage(self.image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
    
    def predict(self):
        # Get the label predictions
        self.net.setInput(self.blob)
        start = time.time()
        self.layerOutputs = self.net.forward(self.ln)
        end = time.time()     
        print("[INFO] Processing image took {:.6f} seconds".format(end-start)) 
    
    def filterFigures(self):
        # Filter figures
        self.boxes = []
        self.confidences = []
        self.classIDs = []
        for output in self.layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.confidence:
                    box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    self.boxes.append([x, y, int(width), int(height)])
                    self.confidences.append(float(confidence))
                    self.classIDs.append(classID)
                    self.idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.confidence, self.threshold)

    
    def labelFigures(self, alert = False):
        # Label figures and draw the frame
        if len(self.idxs) > 0:
            for i in self.idxs.flatten():
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])

                if alert == True and self.labels[self.classIDs[i]] in self.luggage_labels:
                    color = [0, 0, 255]    
                    cv2.putText(self.image, "Alert! Luggage Unattended", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)
                else:
                    color = self.pick_color(index = self.classIDs[i])

                text = "{}: {:.4f}".format(self.labels[self.classIDs[i]], self.confidences[i])
                cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(self.image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    def pick_color(self, index):
        if self.labels[index] in self.luggage_labels:
            return [0, 255, 255]
        if self.labels[index] == "person":
            return [150, 0, 255]
        else:
            return [0]

    def non_overlapping_figures(self):
        self.idxs = self.idxs.flatten()
        self.idxs = np.sort(self.idxs)

        self.classIDs = [self.classIDs[i] for i in self.idxs]
        self.confidences = [self.confidences[i] for i in self.idxs]
        self.boxes = [self.boxes[i] for i in self.idxs]

        return self.classIDs, self.confidences, self.boxes

    def show_picture_obj(self, alert = False):
        # Show the image
        self.labelFigures(alert)
        cv2.imshow("image", self.image)
        cv2.waitKey(0)