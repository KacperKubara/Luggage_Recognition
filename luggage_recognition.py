import cv2
import argparse

from object_recogniser import ObjectRecogniser
from luggage_recogniser import LuggageRecogniser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Luggage Recognition Terminal")
    parser.add_argument('--input_path',
                        help="Relative path to the image for the Luggage Recognition")
    parser.add_argument('--output_path',
                        help="Relative path to the output image for the Luggage Recognition")
    args = parser.parse_args()
  
    luggageRecogniser = LuggageRecogniser()
    print(luggageRecogniser.is_attended())
    luggageRecogniser.draw_line()
    luggageRecogniser.show_picture()
    
    luggageRecogniser.set_picture(make_photo="yes")
    luggageRecogniser.is_attended()
    luggageRecogniser.draw_line()
    luggageRecogniser.show_picture()