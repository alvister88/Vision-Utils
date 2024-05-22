# Yolov8 Modules

## Info: 
Modules for processing Yolov8 model with wandb integration. 

## User-Manual:
For ease of use, copy the same file structure within your own repository and copy the sample code inside of main for each module. 

### Configs
- Files inside of config-templates are yaml files containing the setup(detection.yaml) and the list of tunable hyperparameters(train-params.yaml). 
- The config-examples directory contains and example usage of said files. 
- The config directory is an empty directory that shows the file structure relative to scripts that can be used for easy setup; the modules use relative pathing so keeping the file structure allows you to use the pre-programmed relative paths.

### Weights
- Contains Yolov8 pretrained weights for transfer learning


