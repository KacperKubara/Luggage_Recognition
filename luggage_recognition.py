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
    
    # First functionality test
    luggageRecogniser = LuggageRecogniser()
    print(luggageRecogniser.is_attended())
    luggageRecogniser.draw_line()
    luggageRecogniser.show_picture(alert = True)
    
    # Second functionality test: make photo, alert when unattended
    luggageRecogniser.set_picture(make_photo="no")
    is_attended = luggageRecogniser.is_attended()
    luggageRecogniser.draw_line()
    if is_attended == False:
        luggageRecogniser.show_picture(alert = True)
    else:
        luggageRecogniser.show_picture(alert = False)