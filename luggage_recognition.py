from luggage_recogniser import LuggageRecogniser

if __name__ == "__main__":

    # First functionality test
    luggageRecogniser = LuggageRecogniser()
    
    # Second functionality test: make photo, alert when unattended
    luggageRecogniser.set_picture(make_photo="no", image_path="test_data/luggage0.jpeg")
    luggageRecogniser.set_picture(make_photo="no", image_path="test_data/luggage1.jpeg")
    luggageRecogniser.set_picture(make_photo="no", image_path="test_data/luggage2.jpeg")
    luggageRecogniser.set_picture(make_photo="yes")    
    luggageRecogniser.set_picture(make_photo="yes")