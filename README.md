# Vehicle-Front-Rear-Detection-for-License-Plate-Detection-Enhancement
[Vehicle-Front-Rear-Detection-for-License-Plate-Detection-Enhancement](https://drive.google.com/open?id=1_fjEAgBhQdqUdMInNPVEodBOAX1pyJtq) @ 2019 MITA (International Conference on Multimedia Information Technology and Applications) by Shao-Heng Kuo, Jin-Eun Ok and Eui-Young Cha.
### A network for detecting and classifying vehicle's front part and rear part.
Check our paper for further application on license plate detection by utilizing our model. 
![result_sample](https://user-images.githubusercontent.com/21314064/61183160-11918780-a62d-11e9-9d20-8888df528094.jpg)

# Testing environment
Ubuntu 16.04, python 2.7.16, Keras 2.2.4, tensorflow 1.5.0 with GPU GTX1080

# How to run
This part will contain running a refined version of license plate detection process based on the previous work proposed by Sergio Montazzolli Silva: https://github.com/sergiomsilva/alpr-unconstrained

## 1. Compile darknet
Compile the modified version of darknet in this repository, cd into the darknet directory and type
```
$ make
```
it is possible to modify the makefile in the darknet folder for further configs before compiling, for example, turning off GPU support, which you need to set the GPU=0 in the makefile, default will be GPU supported and turning on the cudnn. If you wonder what is the difference between the origin darknet and the one in our repo, well, we just made it available for reading numpy array.

## 2. Get the models and weights
After compiling the darknet, use the bash script to download all the required models and weights.
```
$ bash get-networks.sh
```
if the download link for our FRD weights is down, check [here](https://drive.google.com/open?id=1O18taeM0wS1kLBTowB64TyHFZmUq1Gxj) to download directly and decompress them to data/FRD/

## 3. Run pipeline_withFRD.py
If everything works well, the script pipeline_withFRD.py will do license plate detection on our sample images, and the results will be in the output folder, some txt files will also be generated with license plate reassignment information.

## 4. To try on your own images
Modify line 26 in pipeline_withFRD.py
```
input_dir = 'samples/overlap_case'
```
let the input_dir be your folder which includes your own images. Run and that's yours!

# How to only utilize the Front-Rear Detection
Run the Front_Rear_Detect.py script, the results will be purely detecting and classifying vehicle's front part and rear part.
![FRD_results](https://user-images.githubusercontent.com/21314064/61181337-a76ce880-a614-11e9-934d-abeb87dfe568.jpg)

# About the dataset for Front-Rear Detection
### 1. Images
We used the Cars Dataset proposed in 3D Object Representations for Fine-Grained Categorization. Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei.<br/>
[dataset download site](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
### 2. Annoations
For the annotation of bounding box in YOLO training format:<br/>
[Training](https://drive.google.com/open?id=1ygqCUyxRPZ5x_6ZsgyZni4RxIeuUkiHq)<br/>
[Testing (Validation)](https://drive.google.com/open?id=1V8XlS4gQt_KD5g02ctXxtZ49IZ1yfHnk)<br/>
The file names refer to corresponding Training/Testing file names in Cars Dataset.

