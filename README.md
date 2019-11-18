# display_calibration

This project trains an artificial neural network (ANN) to calibrate the luminance response of an LCD display
according to the DICOM GSDF.


## Getting Started


### Sub-Modules Organization

The project is splitted in several sub-modules:

* **model-1**: This is a model to predict the measured JND index of the display for each gray level 
* **model-2**:
 This is a model to correct the JND indexes according to the DICOM GSDF
* **model-3**:
 This is a model to predict the expected gray level for each JND value
* **test_pattern_correction**:
 This module calibrates the TG270 test patterns
* **image calibration**:
 This module calibrates a grayscale input image



### Requirements

* tensorflow
* keras
* numpy
* matplotlib.pyplot
* pandas
* PIL
* imageio

### Usage

1. Perform luminance measurements using the TG270-ULN test patterns
2. Insert the values obtained from luminance measurements in the file "measured_luminance.txt" 
3. Run "model-1.py" to train and validate the model
4. Run "model-2.py" to train and validate the model
5. Run "model-3.py" to train and validate the model
6. Run "test_pattern_correction.py" to apply the ANN to calibrate the TG270-ULN test patterns
7. Repeat luminance measurement using the calibrated TG270-ULN test patterns
8. Run "image_calibration.py" to calibrate any grayscale input image


## Author
Djalms S. Santos - Budapest University of Technology and Economics - BME

Department of mechatronics, optics, and mechanical engineering informatics - MOGI

contact: djalma.simoes@mogi.bme.hu


