import torch
from collections import OrderedDict
import torch.nn

model = torch.nn.Sequential(OrderedDict([
    ('bw_conv1_1', torch.nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))),
    ('relu1_1', torch.nn.ReLU()),
    ('conv1_2', torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
    ('relu1_2', torch.nn.ReLU()),
    #
    ('conv1_2norm', torch.nn.BatchNorm2d(64, affine=False)),
    # Batch norm conv1 to add
    ('conv2_1', torch.nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))),
    ('relu2_1', torch.nn.ReLU()),
    ('conv2_2', torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
    ('relu2_2', torch.nn.ReLU()),
    #
    ('conv2_2norm', torch.nn.BatchNorm2d(128, affine=False)),
    # bacth norm conv2 to add
    ('conv3_1', torch.nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))),
    ('relu3_1', torch.nn.ReLU()),
    ('conv3_2', torch.nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))),
    ('relu3_2', torch.nn.ReLU()),
    ('conv3_3', torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
    ('relu3_3', torch.nn.ReLU()),
    ('conv3_3norm', torch.nn.BatchNorm2d(256, affine=False)),

    ('conv4_1', torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))),
    ('relu4_1', torch.nn.ReLU()),
    ('conv4_2', torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))),
    ('relu4_2', torch.nn.ReLU()),
    ('conv4_3', torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))),
    ('relu4_3', torch.nn.ReLU()),
    ('conv4_3norm', torch.nn.BatchNorm2d(512, affine=False)),

    ('conv5_1', torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))),
    ('relu5_1', torch.nn.ReLU()),
    ('conv5_2', torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))),
    ('relu5_2', torch.nn.ReLU()),
    ('conv5_3', torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))),
    ('relu5_3', torch.nn.ReLU()),
    ('conv5_3norm', torch.nn.BatchNorm2d(512, affine=False)),

    ('conv6_1', torch.nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=(2, 2), padding=(2, 2))),
    ('relu6_1', torch.nn.ReLU()),
    ('conv6_2', torch.nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=(2, 2), padding=(2, 2))),
    ('relu6_2', torch.nn.ReLU()),
    ('conv6_3', torch.nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=(2, 2), padding=(2, 2))),
    ('relu6_3', torch.nn.ReLU()),
    ('conv6_3norm', torch.nn.BatchNorm2d(512, affine=False)),

    ('conv7_1', torch.nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1))),
    ('relu7_1', torch.nn.ReLU()),
    ('conv7_2', torch.nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1))),
    ('relu7_2', torch.nn.ReLU()),
    ('conv7_3', torch.nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1))),
    ('relu7_3', torch.nn.ReLU()),
    #
    ('conv7_3norm', torch.nn.BatchNorm2d(512, affine=False)),

    ('conv8_1', torch.nn.ConvTranspose2d(512, 256, kernel_size=(4,4), stride=(2, 2), padding=(1, 1), dilation=(1, 1))),
    ('relu8_1', torch.nn.ReLU()),
    ('conv8_2', torch.nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1))),
    ('relu8_2', torch.nn.ReLU()),
    ('conv8_3', torch.nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1))),
    ('relu8_3', torch.nn.ReLU()),
    ('conv8_313', torch.nn.Conv2d(256, 313, kernel_size=(1, 1), dilation=(1, 1), stride=(1, 1))),
    # Maybe try using multiplication with constant layer
    ('activation', torch.nn.Softmax2d()),
    ('class8_ab', torch.nn.Conv2d(313, 2, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1)))
]))

model.eval()
# print(model)

import pickle
with open('models/model.pkl', 'rb') as f:
    weights = pickle.load(f,encoding='latin1')
W = []
B = []
for i in range(len(weights)):
    if len(weights[i]['weights'])!=0:
#         if len(weights[i]['weights'][0].shape)>1:
#             #print(weights[i]['weights'][0].shape)
#             weights[i]['weights'][0] = rot90(weights[i]['weights'][0])
#         if "norm" in weights[i]['name']:
#             W.append(weights[i]['weights'][0])
#             B.append(weights[i]['weights'][1])
#             continue
        W.append(weights[i]['weights'][0])#.transpose(2,3,1,0))
        B.append(weights[i]['weights'][1])

to_load = [0,2,4,5,7,9,10,12,14,16,17,19,21,23,24,26,28,30,31,33,35,37,38,40,42,44,45,47,49,51,52]

loaded = 0
for i in to_load:
#     print("Weight shape,", model[i].weight.shape)
#     print("loaded shape,", W[loaded].shape)
    model[i].weight = torch.nn.Parameter(torch.from_numpy(W[loaded]), requires_grad=False)
    model[i].bias = torch.nn.Parameter(torch.from_numpy(B[loaded]), requires_grad=False)
    loaded += 1

import skimage.color as color
import skimage.io
from skimage.transform import resize
import numpy as np
import sys

(H_in,W_in) = 224,224
(H_out,W_out) =  56,56

# load the original image
path_img = 'img/lena.png'
if len(sys.argv) > 1:
    path_img = sys.argv[1]
img_rgb = skimage.io.imread(path_img)

if len(img_rgb.shape) == 2:
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

test = img_l_rs.reshape(1,1, 224,224)
res = model.forward(torch.Tensor(test))

res = res[0, :, :, :]
res = res.detach().numpy()

res = np.moveaxis(res, 0, -1)
import scipy.misc
import numpy as np
import scipy.ndimage.interpolation as sni
ab_dec_us = sni.zoom(res,(1.*H_orig/H_out, 1.*W_orig/W_out, 1)) # upsample to match size of original image L
img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
img_rgb_out = (255*np.clip(color.lab2rgb(img_lab_out),0,1)).astype('uint8') # convert back to rgbplt.figure(figsi


path_tores = 'outfile.jpg'
if len(sys.argv) > 2:
    path_tores = sys.argv[2]
scipy.misc.imsave(path_tores, img_rgb_out)
