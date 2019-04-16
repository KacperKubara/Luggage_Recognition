# Luggage_Recognition
Project to recognise unattended luggage in public spaces. 
## Project Structure
* test_data  -> test images with luggage which unattended and attended
* train_data -> train dataset with images of different luggage and people(not currently implemented)
* luggage_recognition.py -> main script demonstrating the functionality
* luggage_recogniser.py  -> class implementing the luggage detection algorithm
* utils.py -> helper function to help maintaining a clear code
## Current Functionality
* Objective 5) but Objective 4) not implemented
## Development objectives
1) Luggage and owner correctly detected:
    * Load image 
    * If luggage with the owner detected return 1
    * If luggage without the owner detecter return 0
    * Luggage is considered as: backpack, suitcase and handbag
2) Distance between luggage and owner is computed
3) Algorithm to determine if the luggage is unattended
4) Add alerts on the picture if unattended luggage is detected
5) Add camera functionality
6) Real-time camera detection