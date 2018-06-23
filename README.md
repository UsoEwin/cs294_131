# CS 294-131 Course Project
## Fiat “Liquet” - Super-Resolution Using Deep Learning

## Jingbo Wu

### Introduction

In this project, our problem definition is to achieve the goal that generating high-resolution images from low-quality inputs using Super-Resolution Convolutional Neural Network (SRCNN), by reimplementing the experiments in the paper, Image Super-Resolution Using Deep Convolutional Networks.

We are interested in the technology of super-resolution because it is both challenging and important. Due to the nature of the project,  we need to infer more information of the graphs from a limited amount of provided information, which is not meant to be tackled with ease. But actually, the challenges indicates the importance of our outcomes.

As an outcome, we used TensorFlow to reimplement the SRCNN model and test it.
### Dependency

	*Tensorflow for training model and computing outputs
	*OpenCV for reading the images and generating outputs
	*h5py for saving checkpoint files

### Train the model

Simply type:
```
python main.py --mode True
```

if you want to see the flag 
```
python main.py -h
```

### Test the mode

If you don't input a Test image, it will be default image
```
python main.py
```
then result will put in the result directory


If you want to Test your own iamge

use `test_img` flag

```
python main.py --test_img [PATH TO YOUR IMAGE]
```

The outcome will be in the result directory with name result.png

## References
	
   [Original Caffe code](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
   
   [Tensorflow-srcnn](https://github.com/liliumao/Tensorflow-srcnn)
