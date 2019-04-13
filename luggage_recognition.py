import cv2
import argparse

from luggage_recogniser import Luggage_Recogniser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Luggage Recognition Terminal")
    parser.add_argument('--input_path',
                        help="Relative path to the image for the Luggage Recognition")
    parser.add_argument('--output_path',
                        help="Relative path to the output image for the Luggage Recognition")
    args = parser.parse_args()
    luggageRecogniser = Luggage_Recogniser(image_path = "test_data/luggage0.jpeg", confidence = 0.5, threshold = 0.3)
    luggageRecogniser.detect_objects()