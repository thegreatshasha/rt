import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
from chainer import cuda 
import cupy 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, transforms 
import random 

class VGGNet(chainer.Chain):

    """
    VGGNet
    - It takes (224, 224, 3) sized image as imput
    """

    def __init__(self):
        super(VGGNet, self).__init__(
            # VGG base network
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_4=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_4=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            
            # Localization relu convolutions (denseboxes)
            reg_conv5_1=L.Convolution2D(768, 512, 1, stride=1, pad=0),
            reg_conv5_2=L.Convolution2D(512, 4, 1, stride=1, pad=0),
            
            # Classification relu convolutions denseboxes
            class_conv5_1=L.Convolution2D(768, 512, 1, stride=1, pad=0),
            class_conv5_2=L.Convolution2D(512, 1,1, stride=1, pad=0),
            
            

        )
        self.train = False

    def __call__(self, x):
        # Conv 1 block VGG
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h_pool0 = F.max_pooling_2d(h, 2, stride=2)
        
        # Conv2 block VGG
        h = F.relu(self.conv2_1(h_pool0))
        h = F.relu(self.conv2_2(h))
        h_pool1 = F.max_pooling_2d(h, 2, stride=2)

        # Conv 3 block VGG
        h = F.relu(self.conv3_1(h_pool1))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h_concat = F.relu(self.conv3_4(h))
        h_pool2 = F.max_pooling_2d(h_concat,2,stride= 2)
        
        # Conv 4 block VGG
        h = F.relu(self.conv4_1(h_pool2))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.relu(self.conv4_4(h))

        h = F.resize_images(h, (h.shape[2]*2,h.shape[2]*2) )
       
        h = F.concat((h, h_concat), axis=1)
       
        # Localization
        h_reg =F.relu(self.reg_conv5_1(h)) 
        h_reg = F.relu(self.reg_conv5_2(h_reg)) 
        
        # Classification
        h_class =F.relu(self.class_conv5_1(h)) 
        h_class = F.relu(self.class_conv5_2(h_class))
        
        return h_class, h_reg
