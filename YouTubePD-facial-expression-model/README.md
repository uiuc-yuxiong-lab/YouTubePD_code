# YouTubePD: A Multimodal Benchmark for Parkinson’s Disease Analysis

# Data
Please add videos to the /data folder and change the .csv files to the following format.

Videos: path + /video{idx}_final.mp4
Bounding Boxes: path + /video{idx}_kpts

# Code
 YouTubePD runs on the open-source PySlowFast framework. Please follow the corresponding instructions for setup. https://github.com/facebookresearch/SlowFast. This is also included in the slowfast folder. 

# Models
The pretrained ResNet50 FER backbone can be found here: https://drive.google.com/file/d/1i-sUHxcDSyXsXnRr1yppbUOzs49l_G8W/view?usp=sharing

# Training
To train or run models, modify the configuration file in slowfast/configs. To run, use python tools/run_net.py --cfg configs/PD/your_config.yaml
