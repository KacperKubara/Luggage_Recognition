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
    """objectRecogniser = ObjectRecogniser(image_path = "test_data/luggage1.jpeg", confidence = 0.5, threshold = 0.3)
    objectRecogniser.is_attended()
    objectRecogniser.show_picture()"""
    luggageRecogniser = LuggageRecogniser()
    print(luggageRecogniser.is_attended())