# -*- coding: utf-8 -*-
# Set the right path to your model file, pretrained model
# and the image you would like to classify.
caffe_root='/home/jane/caffe-master/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import matplotlib.pyplot as plt
import os
import pylab
import numpy as np

class CaffeModel():
    net = None
    def __init__(self, MODEL_FILE, PRETRAINED):
        caffe.set_mode_gpu()
        self.net = caffe.Classifier(MODEL_FILE, PRETRAINED,channel_swap=(2,1,0),raw_scale=299,image_dims=(299, 299))

    def predict(self, file):
        image = caffe.io.load_image(file)
        out = self.net.predict([image])
        return out
