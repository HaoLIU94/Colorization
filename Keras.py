
# coding: utf-8

# # Environnement set up

# In[1]:


from __future__ import division
from math import ceil
import numpy as np
from tensorflow.python.keras.utils import plot_model
from tensorflow.python import keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.layers import Input, Lambda, Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Reshape, ZeroPadding2D, Multiply, Activation
from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.models import save_model
from tensorflow.python.keras import backend as K
import tensorflow as tf
import skimage.color as color
import skimage.io
from skimage.transform import resize
import matplotlib.pyplot as plt
import pickle


# In[2]:


def build_model():

    # Input image format
    image_size = [256,256,1]
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    # Design the actual network
    x = Input(shape=(img_height, img_width, img_channels),name='Input')
#     print(x)
#     x = Lambda(lambda z: z/127.5 - 1., # Convert input feature range to [-1,1]
#                     output_shape=(img_height, img_width, img_channels),
#                     name='lambda1')(x)
#     Reheat = K.placeholder((None,56,56,313))
    print(x)
    conv1 = Conv2D(64, (3, 3), name='bw_conv1_1', strides=(1, 1), padding="same")(x)
    conv1 = ReLU(name='relu1_1')(conv1)
    conv1 = ZeroPadding2D(padding=(1, 1))(conv1)
    conv1 = Conv2D(64, (3, 3), name='conv1_2', strides=(2, 2), padding="valid")(conv1)
    conv1 = ReLU(name='relu1_2')(conv1)
    conv1 = BatchNormalization(name='bn1')(conv1) # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
    
    print(conv1)
    conv2 = Conv2D(128, (3, 3), name='conv2_1', strides=(1, 1), padding="same")(conv1)
    conv2 = ReLU(name='relu2_1')(conv2)
    conv2 = ZeroPadding2D(padding=(1, 1))(conv2)
    conv2 = Conv2D(128, (3, 3), name='conv2_2', strides=(2, 2), padding="valid")(conv2)
    conv2 = ReLU(name='relu2_2')(conv2)
    conv2 = BatchNormalization(name='bn2')(conv2)
    
    print(conv2)
    conv3 = Conv2D(256, (3, 3), name='conv3_1', strides=(1, 1), padding="same")(conv2)
    conv3 = ReLU(name='relu3_1')(conv3)
    conv3 = Conv2D(256, (3, 3), name='conv3_2', strides=(1, 1), padding="same")(conv3)
    conv3 = ReLU(name='relu3_2')(conv3)
    conv3 = ZeroPadding2D(padding=(1, 1))(conv3)
    conv3 = Conv2D(256, (3, 3), name='conv3_3', strides=(2, 2), padding="valid")(conv3)
    conv3 = ReLU(name='relu3_3')(conv3)
    conv3 = BatchNormalization(name='bn3')(conv3)
    
    print(conv3)
    conv4 = Conv2D(512, (3, 3), name='conv4_1', strides=(1, 1),dilation_rate=(1, 1), padding="same")(conv3)
    conv4 = ReLU(name='relu4_1')(conv4)
    conv4 = Conv2D(512, (3, 3), name='conv4_2', strides=(1, 1),dilation_rate=(1, 1), padding="same")(conv4)
    conv4 = ReLU(name='relu4_2')(conv4)
    conv4 = Conv2D(512, (3, 3), name='conv4_3', strides=(1, 1),dilation_rate=(1, 1), padding="same")(conv4)
    conv4 = ReLU(name='relu4_3')(conv4)
    conv4 = BatchNormalization(name='bn4')(conv4)
    
    print(conv4)
    conv5 = ZeroPadding2D(padding=(2, 2))(conv4)
    conv5 = Conv2D(512, (3, 3), name='conv5_1', strides=(1, 1),dilation_rate=(2, 2), padding="valid")(conv5)
    conv5 = ReLU(name='relu5_1')(conv5)
    conv5 = ZeroPadding2D(padding=(2, 2))(conv5)
    conv5 = Conv2D(512, (3, 3), name='conv5_2', strides=(1, 1),dilation_rate=(2, 2), padding="valid")(conv5)
    conv5 = ReLU(name='relu5_2')(conv5)
    conv5 = ZeroPadding2D(padding=(2, 2))(conv5)
    conv5 = Conv2D(512, (3, 3), name='conv5_3', strides=(1, 1),dilation_rate=(2, 2), padding="valid")(conv5)
    conv5 = ReLU(name='relu5_3')(conv5)
    conv5 = BatchNormalization(name='bn5')(conv5)
    
    print(conv5)
    conv6 = ZeroPadding2D(padding=(2, 2))(conv5)
    conv6 = Conv2D(512, (3, 3), name='conv6_1', strides=(1, 1),dilation_rate=(2, 2), padding="valid")(conv6)
    conv6 = ReLU(name='relu6_1')(conv6)
    conv6 = ZeroPadding2D(padding=(2, 2))(conv6)
    conv6 = Conv2D(512, (3, 3), name='conv6_2', strides=(1, 1),dilation_rate=(2, 2), padding="valid")(conv6)
    conv6 = ReLU(name='relu6_2')(conv6)
    conv6 = ZeroPadding2D(padding=(2, 2))(conv6)
    conv6 = Conv2D(512, (3, 3), name='conv6_3', strides=(1, 1),dilation_rate=(2, 2), padding="valid")(conv6)
    conv6 = ReLU(name='relu6_3')(conv6)
    conv6 = BatchNormalization(name='bn6')(conv6)
    
    print(conv6)
    conv7 = Conv2D(512, (3, 3), name='conv7_1', strides=(1, 1), dilation_rate=(1, 1), padding="same")(conv6)
    conv7 = ReLU(name='relu7_1')(conv7)
    conv7 = Conv2D(512, (3, 3), name='conv7_2', strides=(1, 1), dilation_rate=(1, 1), padding="same")(conv7)
    conv7 = ReLU(name='relu7_2')(conv7)
    conv7 = Conv2D(512, (3, 3), name='conv7_3', strides=(1, 1), dilation_rate=(1, 1), padding="same")(conv7)
    conv7 = ReLU(name='relu7_3')(conv7)
    conv7 = BatchNormalization(name='bn7')(conv7)
    
    print(conv7)
    conv8 = Conv2DTranspose(256, (4, 4), name='conv8_1', strides=(2, 2), dilation_rate=(1, 1), padding="same")(conv7)
    conv8 = ReLU(name='relu8_1')(conv8)
    conv8 = Conv2D(256, (3, 3), name='conv8_2', strides=(1, 1), dilation_rate=(1, 1), padding="same")(conv8)
    conv8 = ReLU(name='relu8_2')(conv8)
    conv8 = Conv2D(256, (3, 3), name='conv8_3', strides=(1, 1), dilation_rate=(1, 1), padding="same")(conv8)
    conv8 = ReLU(name='relu8_3')(conv8)
   
    print(conv8)
    conv8 = Conv2D(313, (1, 1), name='conv8_313', strides=(1, 1), dilation_rate=(1, 1), padding="valid")(conv8)
#     print(conv8)
#     print(Reheat)
#     conv8 = Multiply([Reheat,conv8])
    print(conv8)
    softmax = Activation('softmax', name='Softmax')(conv8)
    class_ab = Conv2D(2, (1, 1), name='class8_ab', strides=(1, 1), dilation_rate=(1, 1), padding="valid")(softmax)
    print(class_ab)
    model = Model(inputs=[x], outputs=[class_ab])

    return model


# In[3]:


model = build_model()


# In[4]:


model.summary()


# In[5]:


from keras.utils import plot_model
plot_model(model, to_file='model.png')


# In[6]:


with open('models/model.pkl', 'rb') as f:
    weights = pickle.load(f,encoding='latin1')

    
def rot90(W):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = np.rot90(W[i, j],4)
    return W

W = []
B = []
for i in range(len(weights)):
    if len(weights[i]['weights'])!=0:
        if len(weights[i]['weights'][0].shape)>1:
        #print(weights[i]['weights'][0].shape)
            weights[i]['weights'][0] = rot90(weights[i]['weights'][0])
        if "norm" in weights[i]['name']:
            W.append(weights[i]['weights'][0])
            B.append(weights[i]['weights'][1])
            continue
        W.append(weights[i]['weights'][0].transpose(2,3,1,0))
        B.append(weights[i]['weights'][1])

#Read weights from caffe pickle to keras model
j = 0
for i in range(len(model.layers)):
      if len(model.layers[i].get_weights())!=0:
#     print(i,j)
        if "BatchNormal" in str(model.layers[i]):
            model.layers[i].set_weights([np.zeros(W[j].shape),np.zeros(W[j].shape),W[j],B[j]])
        else:
            model.layers[i].set_weights([W[j],B[j]])
        j+= 1 


# In[7]:


# model.load_weights('./models/model.h5')
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adam)


# In[8]:


model.summary()


# In[9]:


(H_in,W_in) = model.input_shape[1],model.input_shape[2]# get input shape
(H_out,W_out) = model.output_shape[1],model.output_shape[2]  # get output shape

# load the original image
img_rgb = skimage.io.imread('img/NotreDame.png')
if len(img_rgb.shape) ==2:
    img_rgb = np.stack((img_rgb,) * 3, -1)
elif img_rgb.shape[2] != 3:
    img_rgb = np.stack((img_rgb,) * 3, -1)

img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
img_l = img_lab[:,:,0] # pull out L channel
(H_orig,W_orig) = img_rgb.shape[:2] # original image size

# create grayscale version of image (just for displaying)
img_lab_bw = img_lab.copy()
img_lab_bw[:,:,1:] = 0
img_rgb_bw = color.lab2rgb(img_lab_bw)

# resize image to network input size
img_rs = resize(img_rgb,(H_in,W_in)) # resize image to network input size
img_lab_rs = color.rgb2lab(img_rs)
img_l_rs = img_lab_rs[:,:,0]


# In[10]:


# show original image, along with grayscale input to the network
plt.figure(figsize=(20,10))
img_pad = np.ones((H_orig,int(W_orig/10),3))
plt.imshow(np.hstack((img_rgb, img_pad, img_rgb_bw)))
plt.title('(Left) Loaded image   /   (Right) Grayscale input to network')
plt.axis('off');


# In[11]:


test = img_l_rs.reshape(1,H_in,W_in,1)
res = model.predict(test)


# In[12]:


import scipy.ndimage.interpolation as sni
ab_dec_us = sni.zoom(res.reshape([H_out,W_out,2]),(1.*H_orig/H_out, 1.*W_orig/W_out,1)) # upsample to match size of original image L
img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
img_rgb_out = (255*np.clip(color.lab2rgb(img_lab_out),0,1)).astype('uint8') # convert back to rgb
plt.figure(figsize=(20,10))
plt.imshow(img_rgb_out);
plt.axis('off');

import scipy.misc
scipy.misc.imsave('outfile.jpg', img_rgb_out)

