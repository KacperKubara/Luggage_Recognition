import cv2
import utils
import time 
import numpy as np

class Luggage_Recogniser:
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

    def is_attended(self):
        #Algorithm here
        pass

# Return the detected object's labels with their coordinates
    def detect_objects(self):
        self.getLayerNames()
        self.imagePreprocessing()
        self.predict()

        classIDs, confidences, boxes, idxs = self.filterFigures()
        print("ClassID: {} Confidences: {} Boxes: {} Idxs: {}"
        .format(classIDs, confidences, boxes, idxs))
        self.labelFigures(classIDs, confidences, boxes, idxs)
        self.show_picture() 

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
        print("Process took {:.6f} seconds".format(end-start)) 
    
    def filterFigures(self):
        # Filter figures
        boxes = []
        confidences = []
        classIDs = []
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

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence,
	    	        self.threshold)
        print(idxs)
        return (classIDs, confidences, boxes, idxs)
    
    def labelFigures(self, classIDs, confidences, boxes, idxs):
        # Label figures and delte overlapping frames
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [0]
                cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.labels[classIDs[i]], confidences[i])
                cv2.putText(self.image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
                

    def show_picture(self):
        # Show the image
        cv2.imshow("image", self.image)
        cv2.waitKey(0)