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

def normConvolution2D(in_channel, out_channel, ksize=3, stride=1, pad=1):
    return chainer.Chain(
            c  = L.Convolution2D(in_channel, out_channel, ksize=ksize, stride=stride, pad=pad, nobias=True),
            n  = L.BatchNormalization(out_channel, use_beta=False, eps=0.00001),
            b  = L.Bias(shape=[out_channel,]),
    )

def CR(c, h):
    # convolution -> leakyReLU
    h = c.b(c.n(c.c(h)))
    h = F.leaky_relu(h,slope=0.1)
    return h

class VGGNet(chainer.Chain):
    
    

    """
    VGGNet
    - It takes (224, 224, 3) sized image as imput
    """

    def __init__(self):
        super(VGGNet, self).__init__(
            # VGG base network
            conv1_1=normConvolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=normConvolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=normConvolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=normConvolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=normConvolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=normConvolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=normConvolution2D(256, 256, 3, stride=1, pad=1),
            conv3_4=normConvolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=normConvolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=normConvolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=normConvolution2D(512, 512, 3, stride=1, pad=1),
            conv4_4=normConvolution2D(512, 512, 3, stride=1, pad=1),
            
            # Localization relu convolutions (denseboxes)
            reg_conv5_1=normConvolution2D(768, 512, 1, stride=1, pad=0),
            reg_conv5_2=L.Convolution2D(512, 4, 1, stride=1, pad=0),
            
            # Classification relu convolutions denseboxes
            class_conv5_1=normConvolution2D(768, 512, 1, stride=1, pad=0),
            class_conv5_2=normConvolution2D(512, 1,1, stride=1, pad=0),
            
            

        )
        self.train = False

    def __call__(self, x):
        # Conv 1 block VGG
        h = CR(self.conv1_1, x)
        h = CR(self.conv1_2, h)
        h_pool0 = F.max_pooling_2d(h, 2, stride=2)
        
        # Conv2 block VGG
        h = CR(self.conv2_1, h_pool0)
        h = CR(self.conv2_2, h)
        h_pool1 = F.max_pooling_2d(h, 2, stride=2)

        # Conv 3 block VGG
        h = CR(self.conv3_1, h_pool1)
        h = CR(self.conv3_2, h)
        h = CR(self.conv3_3, h)
        h_concat = CR(self.conv3_4, h)
        h_pool2 = F.max_pooling_2d(h_concat,2,stride= 2)
        
        # Conv 4 block VGG
        h = CR(self.conv4_1, h_pool2)
        h = CR(self.conv4_2, h)
        h = CR(self.conv4_3, h)
        h = CR(self.conv4_4, h)

        h = F.resize_images(h, (h.shape[2]*2,h.shape[2]*2) )
       
        h = F.concat((h, h_concat), axis=1)
       
        # Localization
        h_reg =CR(self.reg_conv5_1, h) 
        h_reg = F.sigmoid(self.reg_conv5_2(h_reg)) 
        
        # Classification
        h_class =CR(self.class_conv5_1, h) 
        h_class = CR(self.class_conv5_2, h_class)
        
        return h_class, 2*h_reg-1

    
""" New 8 channel rotation network. let's hook it up immediately """

class DDRNet(chainer.Chain):
    
    """
    DDRNet_net
    - It takes (224, 224, 3) sized image as imput
    - Network has 5 max pooling layers 
    - Network has three upsampling layers
    - Regression classification have 128+512+256+128 =896 filters
    """

    def __init__(self):
        super(DDRNet, self).__init__( 

            #first section
            conv1_1=L.Convolution2D(3, 32, 5, stride=1, pad=1),


            #second section
            conv2_1=L.Convolution2D(32, 64, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            #third section
            conv3_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),


            #fourth section
            conv4_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            #fifth section 
            conv5_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            
            
            #fifth section part 2 
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_4=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_5= L.Convolution2D(512, 128, 1, stride=1, pad=0),


            #regression part
            conv_reg_1= L.Convolution2D(1024, 128, 1, stride=1, pad=0),
            conv_reg_2= L.Convolution2D(128, 8, 1, stride=1, pad=0),

            #classification part 
            conv_class_1= L.Convolution2D(1024, 1, 1, stride=1, pad=0),
    
            
        )
        self.train = False
        
    def __call__(self, x):

        #section 1  
        h = F.relu(self.conv1_1(x))
        h_pool1 = F.max_pooling_2d(h, 2, stride=2) #320/2 = 160 

        #section 2
        h = F.relu(self.conv2_1(h_pool1))
        h = F.relu(self.conv2_2(h))
        h_pool2 = F.max_pooling_2d(h, 2, stride=2) #160/2 = 80
    
        #section 3
        h = F.relu(self.conv3_1(h_pool2))
        h = F.relu(self.conv3_2(h))
        h_pool3 = F.max_pooling_2d(h, 2, stride=2) #80/2 = 80

        #section 4
        h = F.relu(self.conv4_1(h_pool3))
        h = F.relu(self.conv4_2(h))
        h_pool4 = F.max_pooling_2d(h, 2, stride=2) #80/2 = 40
        
        #section 5
        h = F.relu(self.conv5_1(h_pool4))
        h = F.relu(self.conv5_2(h))
        h_pool5 = F.max_pooling_2d(h, 2, stride=2) #40/2 =20 
 
        # section 5 part 2 
        h = F.relu(self.conv5_3(h_pool5))
        h = F.relu(self.conv5_4(h))
        
        h_pool6 = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv5_5(h_pool6))      #20/2 = 10
        
        #upsampling layers 
        h = F.resize_images(h, (h.shape[2]*2,h.shape[2]*2) )
        h = F.concat((h, h_pool5), axis=1)
        #print(h.shape, h_pool5.shape)
        
        h = F.resize_images(h, (h.shape[2]*2,h.shape[2]*2) )
        h = F.concat((h, h_pool4), axis=1)
          
        h = F.resize_images(h, (h.shape[2]*2,h.shape[2]*2) )
        h = F.concat((h, h_pool3), axis=1)
                
        
        h = F.resize_images(h, (h.shape[2]*2,h.shape[2]*2) )

        #regression part 
        h_reg = F.relu(self.conv_reg_1(h))
        h_reg = F.relu(self.conv_reg_2(h_reg))
        h_reg=  F.sigmoid(h_reg)
        h_reg = 2*h_reg-1
        #regression part 
        h_class = F.relu(self.conv_class_1(h))
        
        return h_class, h_reg