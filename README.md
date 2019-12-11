# SSCNN
This is the implement of our paper titled "SSCNN: Superpixel-based Shallow Convolutional Neural Network for Scanned Topographic Map Segmentation"

    
Requirements
1. Matlab (The version I use is 2014b)
2. ShearLib Toolbox for Matlab. (You may copy the ShearLib document to Matlab/toolbox/, and includ the path.)
3. Python3.6
4. pytorch for python (the version I use is v.1.04)
5. pyyaml


Steps to run codes:
1. run Boundary/Boundary_detection.m to genetate the boudnary detection results.
2. run AGWT/line_extraction_COGF.m to generate the linear element image. 
3. run AGWT/GWT_new.m to generate the superpixel. 
4. Please see (or directly run) SCNN/example.py to train/segment.
	please note: the configuration file config.yaml located in SCNN/configs/, you may change the parameters accordingly.

Others:
we put 5 samll STM patches (with manually labeled ground truth) and 1 big STM (without ground truth) in dataset/
