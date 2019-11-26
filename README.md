# display_calibration

This project trains an artificial neural network (ANN) to calibrate the luminance response of an LCD display
according to the DICOM GSDF.


## Getting Started


### Sub-Modules Organization

The project is splitted in several sub-modules:

* **gray_level_to_jnd.py**: Model 1 of the paper. This model predicts the measured JND index for each gray level 
* **jnd_correction.py**:
 Model 2 of the paper. This model predicts the expected JND index (DICOM GSDF)
* **jnd_to_gray_level.py**:
 Model 3 of the paper. This model predicts the expected gray level (DICOM GSDF)
* **test_pattern_correction.py**:
 This module calibrates the TG270 test patterns
* **image calibration.py**:
 This module calibrates a grayscale input image

Folders:

* **TG270 test patterns**: This folder contains the TG270-ULN8 (18 and 52 versions), TG270-pQC, and TG270-sQC test patterns
* **TG270_corrected**: This folder is where the calibrated tests patterns are saved when running test_pattern_correction.py


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
3. Run "gray_level_to_jnd.py" to train and validate model 1*
4. Run "jnd_correction.py" to train and validate model 2*
5. Run "jnd_to_gray_level.py" to train and validate model 3*
6. Run "test_pattern_correction.py" to apply the ANN to calibrate the TG270-ULN test patterns
7. Repeat luminance measurement using the calibrated TG270-ULN test patterns
8. Run "image_calibration.py" to calibrate any grayscale input image

*Uncomment the last line of code to save the model in a HDF5 file

## Author
Djalms S. Santos - Budapest University of Technology and Economics - BME

Department of mechatronics, optics, and mechanical engineering informatics - MOGI

contact: djalma.simoes@mogi.bme.hu


