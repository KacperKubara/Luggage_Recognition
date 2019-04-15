import cv2
import utils
import time 
import numpy as np
from object_recogniser import ObjectRecogniser

class LuggageRecogniser:
    def __init__(self, image_path = "test_data/luggage1.jpeg", confidence = 0.5, threshold = 0.4):
        self.objRecogniser = ObjectRecogniser(image_path, confidence, threshold)
        self.box_person = [0, 0, 0, 0]
        self.box_luggage = [0, 0, 0, 0]

    def is_attended(self):
        # If it is attended returns true, false otherwise
        classIDs, confidences, boxes = self.objRecogniser.detect_objects()
        self.filterBoxes(classIDs, boxes)

        if self.box_person[3] == 0 and self.box_luggage[3] != 0:
            return False
        if self.box_person[3] == 0 and self.box_luggage[3] == 0:
            return True
        if self.box_person[3] != 0 and self.box_luggage[3] == 0:
            return True   
               
        x0, y0, x1, y1 = self.center_cords_all()
        x0_conv, y0_conv = utils.distance_converter(self.box_person[3], 1.8, x0, y0)
        x1_conv, y1_conv = utils.distance_converter(self.box_luggage[3], 0.5, x1, y1)
        self.distance = utils.euclidan_dist(x0_conv, x1_conv, y0_conv, y1_conv)

        self.drawLine(x0, y0, x1, y1)
        self.objRecogniser.show_picture()

        if self.distance > 2: return False
        else: return True

    def filterBoxes(self, classIDs, boxes):
        for i in range(0, len(classIDs)):
            if "person" in self.objRecogniser.labels[classIDs[i]]:
                if self.box_person[2] + self.box_person[3] < boxes[i][2] + boxes[i][3]:
                    self.box_person =  boxes[i]
            if self.objRecogniser.labels[classIDs[i]] in self.objRecogniser.luggage_labels:
                if self.box_luggage[2] +self.box_luggage[3] < boxes[i][2] + boxes[i][3]:
                                        self.box_luggage =  boxes[i]
        return self.box_person, self.box_luggage

    def center_cords_all(self):
        print("Box Person: {} Box Luggage: {}".format(self.box_person, self.box_luggage))
        x0 = utils.center_coords(self.box_person[0], self.box_person[2]) 
        y0 = utils.center_coords(self.box_person[1], self.box_person[3])
        x1 = utils.center_coords(self.box_luggage[0], self.box_luggage[2]) 
        y1 = utils.center_coords(self.box_luggage[1], self.box_luggage[3])
        return (x0, y0, x1, y1)

    def drawLine(self, x0, y0, x1, y1):
        (mX, mY) = (int((x0+x1)/2), int((y0+y1)/2))
        cv2.line(self.objRecogniser.image,(x0, y0), (x1, y1), (255, 0, 0), 5)
        cv2.putText(self.objRecogniser.image, "Distance: {:.2f}m".format(self.distance),
                    (mX, mY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0], 2)    