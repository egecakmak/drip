# <p align="center"> Drip - Dockerized Retraining & Inference Pipeline <p>

<p align="center">
<img src="https://img.shields.io/badge/python-v3.5+-blue">
<img src="https://img.shields.io/badge/version-1.0-green">
<img src="https://img.shields.io/badge/build-passing-green">
<p>

<img src="https://user-images.githubusercontent.com/22041191/64481531-b39f9d00-d1a2-11e9-979b-f9528ba026f0.png" align="right"
     title="Size Limit logo by Ege Cakmak" width="240" height="280">
Drip is a dockerized (as implied by the name) and automatic deep learning environment saving the user from the 
hassle of having to setup the environment. It is for retraining models from the model zoo for retraining an object recognition network.

* Drip is designed to run on NVIDIA GPUs and thereby uses tensorflow-gpu. As a result it includes CUDA 9 and cudNN 9.0 .
* Drip runs all application dependent operations in a docker container it builds automatically. Therefore all you need to install
on the host machine are NVIDIA drivers, Docker and Python.
* Drip aims to be simple to use and time saving.
* Drip also allows users to use the container for operations other than retraining and inference.
* Drip can augment the image dataset to improve its quality.

## Features

* Fully dockerized.
* Written fully on Python 3 and bash.
* Runs only on UNIX for now however the docker container could be deployed on Windows as well.
* Drip does the following automatically during retraining.
  - Sets the environment required.
  - Picks images randomly for evaluation and training.
  - Augments the images with albumentations to improve quality of the dataset. These augmentations can be changed by the user as they wish.
  - Sets up the deep learning network automatically.
  - Generates the required tfrecord files automatically.
  - Starts training.
* Provides a folder with a systematic structure allowing user to access the results easily.
* Allows users to bring their own model and retrain them.
* Comes with TensorBoard to allow users monitor the retraining.
* It can also do inference and save the results.
* All input and output of metadata are completely in JSON.
* Provides the coordinates of detection boxes on the augmented images.

## How It Works

1. Drip works as a pipeline of 8 modular Python files and couple other bash files.
2. 2 of these Python files are like a wrapper and it starts the container. The remaining files run inside the container.

## Requirements
- Docker
- Python 3
- NVIDIA Drivers

## Usage

<details><summary><b>Show instructions for Retraining</b></summary>

1. Install Python, Docker and NVIDIA drivers.

2. Download cudnn-9.0-linux-x64-v7.5.1.10.tgz from https://developer.nvidia.com/cudnn and place it under the files directory.

3. Get a model from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md, rename the zip file into network.zip and place it under the files directory.

4. Edit the class_map.pbtxt so that it has your categories.

5. Prepare a JSON file structured like the provided metadata.json for the metadata.

6. Run the following command to start. Your images need to be placed in a folder and this folder should be place in another. Its structure should be like in the folder images under the files directory.
    ```
    sudo python3 training_starter.py 
    ```
    
    Below are the arguments training_starter.py takes. <br>
    ```
    [--src_json_path SRC_JSON_PATH] Allows user specify the path of the input metadata. default='./files/metadata.json' <br>
    [--image_folder_path IMAGE_FOLDER_PATH] Allows user specify the path of the images. default='./files/images/' <br>
    [--eval_percentage EVAL_PERCENTAGE] Allows user set a percentage of pictures to be chosen for evaluation default=20 <br>
    [--resume_training] Allows jumping right onto the retraining. This requires a session file with the name 'dl' to be placed under the directory pipeline. <br> 
    [--session_folder SESSION_FOLDER] Allows user specify a path for the folder 'dl'. default='./usersession/' <br>
    [--batch_size BATCH_SIZE] Allows user set the batch size that will be used for retraining. default=4 <br>
    [--number_of_steps NUMBER_OF_STEPS] Allows usere set the number of steps while retraining. default=200000 <br>
    ```
    Once Drip is done it will save the results for each folder under a folder called results in the files directory.
    
    Drip will keep you informed about the operations it does. It will also mount a folder called pipeline. In this folder you may find the following folders.
      - dl - Includes all files specific to that retraining session.
      - trash - Includes old session files. Drip will automatically move existing old session folders to this folder if the user tries to start a new session to prevent overwrites.
      - images_augmented - Includes the images that are augmented.
</details>

<details><summary><b>Show instructions for Inference</b></summary>

1. Install Python, Docker and NVIDIA drivers.

2. Download cudnn-9.0-linux-x64-v7.5.1.10.tgz from https://developer.nvidia.com/cudnn and place it under the files directory.

3. Place your frozen inference graph under the directory files and rename it to graph.pb

4. Edit the class_map.pbtxt so that it has your categories.

5. Run the following command to start. Your images need to be placed in a folder and this folder should be placed in another. Its structure should be like in the folder images under the files directory.
    ```
    sudo python3 training_starter.py 
    ```
    ```
    Below are the arguments inference_starter.py takes. <br>
    [--images_path IMAGES_PATH] Allows user specify the path of the images. default='./files/images' <br>
    [--threshold THRESHOLD] Allows user to set a threshold percentage for inference. default=60
    ```
    
    Drip will keep you informed about the operations it does. It will also mount a folder called pipeline. In this folder you may find the following folders.
      - dl - Includes all files specific to that retraining session.
      - trash - Includes old session files. Drip will automatically move existing old session folders to this folder if the user tries to start a new session to prevent overwrites.
      - images_augmented - Includes the images that are augmented.
    
</details>

## Additional Configuration
  - Users may pause the retraining and change the pipeline.config file under the directory pipeline/dl/network_folder_name.
  After change the file user needs to restart the training_starter.py.
  
  - Users can change the augmentations and their probability of getting applied. To do so, user needs to import the packages at line 9, for the augmentations they would like to add. They also need to add these augmentations under the function get_random_aug in the same file.


## Contributions
  Baris Bagcilar

## Thanks 
  Huge thanks to Mr. Dogukan Altay of Selvi Technology for their guidance and help.
  
## Contact
  Ege Cakmak - cakmake@my.yorku.ca

## Some more stuff.
  This project took me a good amount of time but I learned a lot while working on it. If it helps you in any way and just a thank you is enough. Donations are not accepted. Thank you. <br>
  This project is licensed under the Apache license.
