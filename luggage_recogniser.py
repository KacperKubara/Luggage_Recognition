import cv2
import utils

class Luggage_Recogniser:
    def __init__(self, image_name):
        self.image_name = image_name
        self.image = utils.load_image(image_name)

    def is_attended(self):
        #Algorithm here
        pass
        
    def show_picture(self):
        pass