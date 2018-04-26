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
    # readin corresponding data, train or test

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
    #generate small data pieces
    input_seq = []
    lbl_seq = []
    for i in range(len(data)):
        if config.mode:
            inputfig, label, = preprocess(data[i], config.scale)
        else:
            inputfig, label, = preprocess(data[i], config.scale) 
        
        # check the channel numbers
        if len(inputfig.shape) == 3: 
            h, w, c = inputfig.shape
        else:
            # check the channel numbers
            h, w = inputfig.shape 
        # counting the sub imgs
        nx, ny = 0, 0
        for x in range(0, h - config.image_size + 1, config.stride):
            nx += 1 
            ny = 0
            for y in range(0, w - config.image_size + 1, config.stride):

                ny += 1

                sub_input = inputfig[x: x + config.image_size, y: y + config.image_size]
                sub_label = label[int(x + padding)+1: int(x + padding + config.label_size)+1, int(y + padding)+1: int(y + padding + config.label_size)+1]

                # reshape the subinput and sublabel
                sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])
                sub_label = sub_label.reshape([config.label_size, config.label_size, config.c_dim])
                # normialize
                sub_input = sub_input/255.0
                sub_label = sub_input/255.0
                

                input_seq.append(sub_input)
                lbl_seq.append(sub_label)

    return input_seq, lbl_seq, nx, ny



def make_data_hf(input_, label_, config):
    # generate the h file
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
    # generate output img
    height, width = images.shape[1], images.shape[2]
    
    img = np.zeros((height*size[0], width*size[1], c_dim))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * height : j * height + height,i * width : i * width + width, :] = image
    return img

def input_setup(config):
    # load the input image and make sub imgs, save them in h5
    data = load_data(config.mode, config.test_img)

    padding = abs(config.image_size - config.label_size) / 2 

    # make sub_input and sub_label, if mode false more return nx, ny
    sub_input_sequence, sub_label_sequence, nx, ny = make_sub_data(data, padding, config)

    # construct array
    arrinput = np.asarray(sub_input_sequence) 
    arrlabel = np.asarray(sub_label_sequence) 

    make_data_hf(arrinput, arrlabel, config)

    return nx, ny

