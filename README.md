LicensePlateSpanish_FindContours_OCRScratch

A project that detects Spanish car license plate numbers using YOLO for plate recognition, `cv2.findContours` for character prediction, and a basic OCR implementation built from scratch, all while utilizing minimal filtering and preprocessing.

Installation:

Download and extract the project to your local drive.

Testing:

Run the program:

`GetNumberSpanishLicensePlate_FindContours_OCRScratch`

The output displays the recognized license plates and the filters that successfully identified them. The image filenames correspond to the license plate numbers, allowing for verification.

The final result shows 16 successful detections out of 21 processed images.

Processing time per image is approximately 1 second.

By changing the folder path on line 15 of the program, you can test any folder; however, the images must be of Spanish license plates and named to match the plate number.

Line 9 contains the parameter `SwOptionPlot="N"`; changing this to "Y" allows you to visualize the character contouring process before the characters are sent to the OCR. The car license plate recognition model, `best.pt`, was developed as part of this project: https://github.com/ablanco1950/LicensePlate_Yolov8_Filters_PaddleOCR

The OCR model (`OCR11Hits.weights.h5`) was trained using the `TrainOCR.py` script from this project: https://github.com/pragatiunna/License-Plate-Number-Detection/tree/main

Modifications were made to the script, as was the case with `OCRScratch_V1.py`.

The training dataset was obtained from https://data.mendeley.com/datasets/nx9xbs4rgx/2; it also appears as `data.zip` in the aforementioned project.

Training was repeated several times; given the speed of the process, it can be run frequently on a personal computer until an optimal model is found—though this implies a degree of overfitting.

References:

https://github.com/pragatiunna/License-Plate-Number-Detection/tree/main

https://data.mendeley.com/datasets/nx9xbs4rgx/2

https://github.com/ablanco1950/LicensePlate_Yolov8_Filters_PaddleOCR

https://gist.github.com/endolith/255291#file-parabolic-py

https://medium.com/@garciafelipe03/image-filters-and-morphological-operations-using-python-89c5bbb8dca0

https://blog.katastros.com/a?ID=01800-4bf623a1-3917-4d54-9b6a-775331ebaf05
