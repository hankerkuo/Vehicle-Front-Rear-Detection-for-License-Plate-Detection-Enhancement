# Vehicle-Front-Rear-Detection-for-License-Plate-Detection-Enhancement
Vehicle-Front-Rear-Detection-for-License-Plate-Detection-Enhancement @ 2019 MITA (International Conference on Multimedia Information Technology and Applications)
![result_sample](https://user-images.githubusercontent.com/21314064/61183160-11918780-a62d-11e9-9d20-8888df528094.jpg)

# Testing environment
Ubuntu 16.04, python 2.7.16, Keras 2.2.4, tensorflow 1.5.0 with GPU GTX1080

# How to run
This part will contain running a refined version of license plate detection process based on previos work proposed by Sergio Montazzolli Silva: https://github.com/sergiomsilva/alpr-unconstrained

## 1. Compile darknet
compile the modified version of darknet in this repository, cd into the darknet directory and type
```
$ &&make
```
it is possible to modify the makefile in the darknet folder for further configs before compiling, for example, turning off GPU support, which you need to set the GPU=0 in the makefile, default will be GPU supported and turning on the cudnn. 

## 2. Run pipiline_withFRD.py
if everything works well, this script will do license plate detection on our sample images, and the results will be in the output folder, some txt files will also be generated with license plate reassignment information.

## 3. To try on your own images
modify line 26 in pipiline_withFRD.py
```
input_dir = 'samples/kr'
```
let the input_dir be your folder which includes your own images. Then run it!

# How to only utilize the front-rear detection
run the FRD.py script in the src folder, the results will be purely detecting and classifying vehicle's front part and rear part.
![FRD_results](https://user-images.githubusercontent.com/21314064/61181337-a76ce880-a614-11e9-934d-abeb87dfe568.jpg)

