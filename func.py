import cv2
import numpy as np
import os 
import glob
import h5py
import tensorflow as tf

def imsave(image, path, config):
    # convert the img to be normial and save it
    if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.result_dir))   
    cv2.imwrite(os.path.join(os.getcwd(),path),image * 255.)


def remove_corner(img, scale =3):
    # check the channel number of figure and remove the remainder of img
    if len(img.shape) ==3:
        height, weight, _ = img.shape
        height = (height / scale) * scale
        weight = (weight / scale) * scale
        img = img[0:int(height), 0:int(weight), :]
    else:
        height, weight = img.shape
        height = (height / scale) * scale
        weight = (weight / scale) * scale
        img = img[0:int(height), 0:int(weight)]
    return img

def checkpoint_dir(config):
    #check the h5 file for different purpose, train or test
    if config.mode:
        return os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    else:
        return os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

def preprocess(path ,scale = 3):
    #preprocessing includes scale up the img using bicubic algo and resize the img for test
    img = cv2.imread(path)
    data_lb = remove_corner(img, scale)
    bicbuic_img = cv2.resize(data_lb,None,fx = 1.0/scale ,fy = 1.0/scale, interpolation = cv2.INTER_CUBIC)
    input_data = cv2.resize(bicbuic_img,None,fx = scale ,fy=scale, interpolation = cv2.INTER_CUBIC)
    # make the LR data
    return input_data, data_lb

def prepare_data(dataset="Train",Input_img=""):
    """
        Args:
            dataset: choose train dataset or test dataset
            For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp',..., 't99.bmp']
    """
    if dataset == "Train":
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp")) # make set of all dataset file path
    else:
        if Input_img !="":
            data = [os.path.join(os.getcwd(),Input_img)]
        else:
            data_dir = os.path.join(os.path.join(os.getcwd(), dataset), "Set5")
            data = glob.glob(os.path.join(data_dir, "*.bmp")) # make set of all dataset file path
    print(data)
    return data

def load_data(mode, test_img):
    if mode:
        data = prepare_data(dataset="Train")
    else:
        if test_img != "":
            return prepare_data(dataset="Test",Input_img=test_img)
        data = prepare_data(dataset="Test")
    return data

def make_sub_data(data, padding, config):
    """
        Make the sub_data set
        Args:
            data : the set of all file path 
            padding : the image padding of input to label
            config : the all flags
    """
    sub_input_sequence = []
    sub_label_sequence = []
    for i in range(len(data)):
        if config.mode:
            input_, label_, = preprocess(data[i], config.scale) # do bicbuic
        else: # Test just one picture
            input_, label_, = preprocess(data[i], config.scale) # do bicbuic
        
        if len(input_.shape) == 3: # is color
            h, w, c = input_.shape
        else:
            h, w = input_.shape # is grayscale

        nx, ny = 0, 0
        for x in range(0, h - config.image_size + 1, config.stride):
            nx += 1; ny = 0
            for y in range(0, w - config.image_size + 1, config.stride):
                ny += 1

                sub_input = input_[x: x + config.image_size, y: y + config.image_size] # 33 * 33
                sub_label = label_[int(x + padding)+1: int(x + padding + config.label_size)+1, int(y + padding)+1: int(y + padding + config.label_size)+1] # 21 * 21


                # Reshape the subinput and sublabel
                sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])
                sub_label = sub_label.reshape([config.label_size, config.label_size, config.c_dim])
                # Normialize
                sub_input =  sub_input / 255.0
                sub_label =  sub_label / 255.0
                

                # Add to sequence
                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

        
    # NOTE: The nx, ny can be ignore in train
    return sub_input_sequence, sub_label_sequence, nx, ny


def read_data(path):
    """
        Read h5 format data file

        Args:
            path: file path of desired file
            data: '.h5' file format that contains  input values
            label: '.h5' file format that contains label values 
    """
    with h5py.File(path, 'r') as hf:
        input_ = np.array(hf.get('input'))
        label_ = np.array(hf.get('label'))
        return input_, label_

def make_data_hf(input_, label_, config):
    """
        Make input data as h5 file format
        Depending on "mode" (flag value), savepath would be change.
    """
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(),config.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.checkpoint_dir))

    if config.mode:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/test.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('input', data=input_)
        hf.create_dataset('label', data=label_)

def merge(images, size, c_dim):
    """
        images is the sub image set, merge it
    """
    h, w = images.shape[1], images.shape[2]
    
    img = np.zeros((h*size[0], w*size[1], c_dim))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h : j * h + h,i * w : i * w + w, :] = image
    return img

def input_setup(config):
    """
        Read image files and make their sub-images and saved them as a h5 file format
    """
    # Load data path, if mode False, get test data
    data = load_data(config.mode, config.test_img)

    padding = abs(config.image_size - config.label_size) / 2 

    # Make sub_input and sub_label, if mode false more return nx, ny
    sub_input_sequence, sub_label_sequence, nx, ny = make_sub_data(data, padding, config)

    # Make list to numpy array. With this transform
    arrinput = np.asarray(sub_input_sequence) # [?, 33, 33, 3]
    arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 3]

    make_data_hf(arrinput, arrlabel, config)

    return nx, ny

