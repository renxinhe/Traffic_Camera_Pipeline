# Traffic Camera Pipeline
Berkeley AUTOLab

## Overview
With the rise of live video streaming, a massive amount of data flows through the internet without getting collected. On the other hand, the recent advance in deep learning has shown the importance of big data. This repo features a pipeline for live stream video collection, object recognition in videos, and prepares the data to be consumed by a 2D intersection traffic simulator.

## Dependencies
See requirements.txt for the complete list of dependencies.

To install all dependencies with pip, run the command
```bash
pip install -r requirements.txt
```

## Pre-trained Weights Download
We recommend placing all model weights in a new directory named `Checkpoint`. Model weight variables can be set under `src/tcp/configs/*_config.py`.

### [Single Shot MultiBox Detector (SSD)](https://github.com/balancap/SSD-Tensorflow)

TCP uses SSD 300 to detect vehicles and pedestrians. To obtain pre-trained weights, download [SSD-300 VGG-based](https://drive.google.com/file/d/0B0qPCUZ-3YwWUXh4UHJrd1RDM3c/view?usp=sharing) from the [SSD-Tensorflow repository](https://github.com/balancap/SSD-Tensorflow#evaluation-on-pascal-voc-2007). Afterwards, unzip `VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt.zip`, and place the contents (\*.ckpt.index and \*.ckpt.data-00000-of-00001) under `Checkpoint/SSD_checkpoint`.

The `self.ssd_checkpoint_path` variable in config file should point to the "ckpt" file of the SSD model.

### [Real-time Recurrent Regression Tracker (Re3)](https://gitlab.com/danielgordon10/re3-tensorflow)

TCP uses Re3 to track vehicles and pedestrians. To obtain pre-trained weights, download the [model](https://gitlab.com/danielgordon10/re3-tensorflow#model) from the official Re3 repo. Place the unzipped contents (model.ckpt-0.\*) under `Checkpoint/Re3_checkpoint`.

The `self.re3_checkpoint_dir` variable in config file should point to the directory where previously unzipped Re3 model files are placed.
