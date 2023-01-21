# Pothole Severity Estimation

![demo](figure/scene01.gif)


## Overview
This project proposes an intelligent system that detects potholes, quantifying and delineate their shape in the 2D space; and estimate the depth information in the 3D space.

![system](figure/Theme_2_model.png)


## Dataset
For generating the training bounding boxes and segmentation masks, we annotate some of the  given images using Roboflow: https://roboflow.com/. The dataset is available in the Data folder.

## Inference


## To run

1. Clone the  model.
<code>https://github.com/LailaMB/Pothole_Severity_Estimation.git</code>

2. Run the <code> python segment/predict.py --weights yolov7-seg.pt --source Theme2_seg_data/input.mp3 --name coco</code>.
3. Run <code>  </code>
