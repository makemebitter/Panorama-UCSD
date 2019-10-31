# Panorama
Source code for project [Panorama](https://adalabucsd.github.io/panorama.html). It is a domain-agnostic video analytics system designed for mitigating the unbounded vocabulary problem. Please check our [tech report](https://adalabucsd.github.io/papers/TR_2019_Panorama.pdf) for more details.

# Prerequisites
1. Python, Keras and TensorFlow. Only tested with Python 2.7, Keras==2.1.4 and tensorflow==1.4.0 on both Ubuntu 16.04.3 LTS. The deployment demos below are also tested with with Keras==2.2.4 and tensorflow==1.12 on OS X 10.14.5 (Mojave). ```ffmpeg``` is also needed for video processing.
2. The requirements can be installed by
	```
    pip install -r requirements.txt
    ```
    **OS X users**: you need to change ```tensorflow-gpu``` to ```tensorflow``` in the ```requirements.txt```.
3. An existing reference model that is capable of object detection and fine-grained classification. Alternatively, a fully annotated (bounding boxes, labels) dataset targeting your application can be used.
4. A CUDA-enabled GPU is highly recommended.

# Quick start with pre-trained weights
We provide an example of deploying Panorama on face recognition tasks. You can download the pre-trained weights and relevent config files [here](https://drive.google.com/file/d/1zkQXVGfMW3HPozQE1Lp8xP2GsUfvsGft/view?usp=sharing). The tarball contains three files:
1. ```faces_config.json``` The Panorama configuration file generated during training.
2. ```panorama_faces_original_loss_weights.h5``` The PanoramaNet weights.
3. ```panorama_faces_original_loss_weights.csv``` The model qualification file required to configure PanoramaNet's cascade processing.

Put these files under folder ```.../Panorama-UCSD/trained_models``` (create the folder if not exists). 

Run the video demo simply by
```
cd panorama/examples
python demo.py
```
Panorama will then start to detect faces and you can see the video feed with bounding boxes. 
1. Now press ```s``` to enter annotation mode. A freezed image will pop up.
2. Move your mouse to the bounding box that you intend to annotate. The box will change color as you hover upon it.
3. Click the box and the program will prompt you for label. Input the label and press enter. Repeate to label other objects on the image window. Once finished press ```c``` to exit annotation mode.
4. The video will resume playing. Panorama's vocabulary is now enlarged to recognize the people's identities. No CNN retraining happened during this process.
5. Press ```q``` to quit.

[![Click for full video](/panorama/examples/panorama_optimized.gif?raw=true)](https://youtu.be/KHoOa-ilaRE)

# End-to-end example with faces
This is an example of training and deploying Panorama on face recognition tasks, as described in our paper.
1. Prepare a long-enough video (~50 hrs) from your video stream. We used 
CBSN (https://www.cbsnews.com/live/) in our paper for face recognition tests.
Create a directory for storing the data by:
	```bash
	$ mkdir -p dataset/faces/raw
	$ mkdir -p dataset/faces/video
	```
	Then put your video under ```dataset/faces/video```.
1. Unpack the video into frames and deploy you reference model on the frames to get
weekly supervised data. You can use the same reference model we used, which is [MTCNN + FaceNet](https://github.com/davidsandberg/facenet). These steps are described in Section 4.3 of our paper.
    1. First create a dir for storing model weights.
        ```
        $ mkdir -p trained_models/align
        ```
    
    1. Then go to [link](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk), download the weights for facenet, unzip, put the extracted folder under ```trained_models```.
    1. Download the weights for MTCNN. Go to [link](https://github.com/davidsandberg/facenet/tree/master/src/align) and download all three ```*.npy``` files to ```trained_weights/align```.
    1. Generate data by running ```panorama/data/generate_data.sh```. You may need to modify the paths in this script.
       ```
       $ cd panorama/data
       $ ./generate_data.sh
       ```
       This will take several hours.
1. Start training by going to ```panorama/examples``` and executing ```run.sh```. The hyperparameters are tunned for this specific example. Depending on your hardware, the training can take up to several days. (~ 2d on a single GTX 1080Ti). This script will also look at your trainning data and generate a config file named ```faces.json```, which we will need later.
1. Once the training is done. We now move on to the next step of configuring the cascade, as described in the paper. The script to do this is ```panorama/example/model_qualification.sh```.
1. Panorama is now ready for deploying. Please check ```examples/examples.ipynb``` for, well, examples.
