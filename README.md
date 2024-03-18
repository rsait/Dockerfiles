# Dockerfiles

Instructions for working with some docker images that have already been uploaded to [DockerHub](https://hub.docker.com/search?q=jmartinezot). The main idea is to provide an easy way of instaling and testing common artificial intelligence applications. Guidelines for contributing to the repository are also provided.

# Table of Contents
1. [Overview](#Overview)
2. [YOLO tracking](#YOLO_tracking)
3. [Contributing](#Contributing)

## Overview

A Dockerfile is a blueprint for building docker images, in the same sense that a computer program is a blueprint for creating executables. DockerHub is a place to host images that can be downloaded without the need for building images from Dockerfiles locally.

In this repository you will find Dockerfiles for local building as well as links to hosted images for running them just by pulling them from the DockerHub repository.

The typical workflow is as follows:

* Check which docker image you are interested in, based on the descriptions in this README file.
* Pull the image from [DockerHub](https://hub.docker.com/search?q=jmartinezot). As an example, We will pull the `test` image.
```bash
docker pull jmartinezot/test
```
* To run the image, you could check the instructions on the corresponding Dockerfile, that are also copied in the corresponding section of this README file. Checking `test.docker` under the `test` subdirectory in this repository, we see that its content is the following:

```bash
# To build the image:
# docker build -f test.docker -t test .

# To create and run a container:
# docker run -it --rm test

FROM ubuntu:22.04
```

and therefore we could run the image and create a container in this way:

```bash
docker run -it --rm test
```

The instructions in the Dockerfile will be adapted to the characteristics of each particular image. Check the Dockerfile for examples of how to use the container.

## YOLO tracking

This image implements YOLO tracking as in the repository [Yolov7 + StrongSORT with OSNet](https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet)

First [YOLOv7](https://github.com/WongKinYiu/yolov7) is used to detect objects from 80 different classes as in the [COCO dataset](https://cocodataset.org)

```python
coco_labels = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}
```
And then [StrongSORT](https://github.com/dyhBUPT/StrongSORT) is used to track the objects.

To pull the image:

```bash
docker pull jmartinezot/yolo_tracking
```

The container can be created and run with 

```
docker run -it --rm --gpus all --shm-size=8g -e DISPLAY=unix$DISPLAY --device /dev/video0 --mount type=bind,source=$(echo $HOME)/yolo_tracking_tmp,target=/tmp --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix yolo_tracking /bin/bash -c 'cat yolo_tracking.readme.txt; bash'
```

Once inside the container, some examples could be:
* Perform tracking of a file included in the image: 
```bash
python track_with_classes.py --source basketball.mp4 --save-tracking --save-vid
```
* Perform tracking of the webcam
```bash
python track_with_classes.py --source 0 --save-tracking --save-vid
```

The script `track_with_classes` is a modification of the original `track.py` provided in the source tracker repository. That file is also present in this image.

Results are saved in the directory "/tmrp/runs", with is mounted on "~/yolo_tracking_tmp" with the given command.

There are three types of results:

* Annotated video withe the bounding boxes, the object class and the detection confidence.
* A csv file with the same information.
* A pickle file with a dataframe containing the same info too.

More details about how to use the tracker can be found in the original [Yolov7 + StrongSORT with OSNet](https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet) repository. 


## Contributing

### Uploading to DockerHub

To upload the `test` image, first tag it accordingly:

```bash
docker tag test:latest jmartinezot/test:latest
```

login to DockeHub
```bash
docker login
```

and then upload it 
```bash
docker push jmartinezot/test:latest
```

