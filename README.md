# Pothole Severity Estimation

![demo](figure/scene01.gif)


## Overview
This project proposes an intelligent system that detects potholes, quantifying and delineate their shape in the 2D space; and estimate the depth information in the 3D space.

![system](figure/Theme_2_model.png)


## Dataset

Trento and Civezzano UAV datasets are not included in the repository, as they are owned by other parties. However, you can try the code with <a href="https://bigearth.eu/datasets.html"> UC-Merced Multilabel Dataset</a> or <a href="https://github.com/Hua-YS/AID-Multilabel-Dataset">AID Multilabel Dataset</a>.

## To run

1. Install the required libraries.
2. Clone the data-efficient transformers (deit) model.
<code> !git clone https://github.com/facebookresearch/deit.git </code>

2. Replace the <code> 'deit/models.py'</code> file with the <code> models.py </code> file included in the repository.
3. Run <code> UAV_image_multilabeling_Transformer.py </code>
