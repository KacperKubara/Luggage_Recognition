import time 
import utils

import cv2
import numpy as np

from object_recogniser import ObjectRecogniser

class LuggageRecogniser(ObjectRecogniser):
    def __init__(self, image_path = "test_data/luggage1.jpeg", video_path = "None", confidence = 0.5, threshold = 0.4):
        super().__init__(image_path, confidence, threshold)
        self.camera = cv2.VideoCapture(0)

    def is_attended(self):
        # If luggage is attended returns true, false otherwise
        classIDs, confidences, boxes = super().detect_objects()
        self.filterBoxes(classIDs, boxes)
        # box[3] == 0 if it doesn't exist 
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
                
        # Luggage is not attended when there is nobody in the 2m vicinity
        if self.distance > 2: return False
        else: return True

    def filterBoxes(self, classIDs, boxes):
        self.box_luggage = [0, 0, 0, 0]
        self.box_person = [0, 0, 0, 0]
        for i in range(0, len(classIDs)):
            if "person" in self.labels[classIDs[i]]:
                if self.box_person[2] + self.box_person[3] < boxes[i][2] + boxes[i][3]:
                    self.box_person =  boxes[i]
            if self.labels[classIDs[i]] in self.luggage_labels:
                if self.box_luggage[2] + self.box_luggage[3] < boxes[i][2] + boxes[i][3]:
                    self.box_luggage =  boxes[i]

    def center_cords_all(self):
        x0 = utils.center_coords(self.box_person[0], self.box_person[2]) 
        y0 = utils.center_coords(self.box_person[1], self.box_person[3])
        x1 = utils.center_coords(self.box_luggage[0], self.box_luggage[2]) 
        y1 = utils.center_coords(self.box_luggage[1], self.box_luggage[3])
        return (x0, y0, x1, y1)

    def draw_line(self):
        if self.box_person[2] != 0 and self.box_luggage[2] != 0:
            x0, y0, x1, y1 = self.center_cords_all()
            (mX, mY) = (int((x0+x1)/2), int((y0+y1)/2))
            cv2.line(self.image,(x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.putText(self.image, "Distance > {:.2f}m".format(self.distance),
                        (mX, mY + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [200, 0, 0], 2)    
    
    def show_picture(self, alert = False):
        super().show_picture_obj(alert)
    
    def set_picture(self, make_photo = "no", image_path = "test_data/luggage2.jpeg"):
        if make_photo == "no":
            self.image_path = image_path
            self.image = utils.load_image(image_path)
        if make_photo == "yes":
            ret, image_camera = self.camera.read()
            self.camera.release()
            self.image = image_camera
    
    def set_video(self, video_path = ""):
        if video_path != "":
            while True:
                vstream = cv2.VideoCapture(video_path)
                (grabbed, image) = vstream.read()
                if grabbed is None:
                    break
                
