import cv2
import numpy as np
import sys
import argparse

def colorize(path_img,path_tores):
	# Specify the paths for the model files 
	protoFile = "model/colorization_deploy_v2.prototxt"
	weightsFile = "model/colorization_release_v2.caffemodel"
	# Read the input image
	# load the original image
	frame = cv2.imread(path_img)
	W_in = 224
	H_in = 224
	# Read the network into Memory 
	net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile) 
	# Load the bin centers
	pts_in_hull = np.load('model/pts_in_hull.npy')
	# populate cluster centers as 1x1 convolution kernel
	pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
	net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
	net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]
	#Convert the rgb values of the input image to the range of 0 to 1
	img_rgb = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
	img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
	img_l = img_lab[:,:,0] # pull out L channel
	# resize the lightness channel to network input size 
	img_l_rs = cv2.resize(img_l, (W_in, H_in)) # resize image to network input size
	img_l_rs -= 50 # subtract 50 for mean-centering
	net.setInput(cv2.dnn.blobFromImage(img_l_rs))
	ab_dec = net.forward()[0,:,:,:].transpose((1,2,0)) # this is our result

	(H_orig,W_orig) = img_rgb.shape[:2] # original image size
	ab_dec_us = cv2.resize(ab_dec, (W_orig, H_orig))
	img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
	img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)

	cv2.imwrite(path_tores, cv2.resize(img_bgr_out*255,(frame.shape[1],frame.shape[0])))

def main(argv):
	#Default illustrution 
	path_img = './img/boat512.png'
	path_tores = 'boat_color.jpg'
	if len(sys.argv) == 3:
	    path_img = sys.argv[1]
	    path_tores = sys.argv[2]
	colorize(path_img,path_tores)
	print("Successfully colorized "+path_img)
	print("Saved as "+path_tores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Script using OpenCV to load Colorization caffe model '
                                                  'Example : python OpenCV.py input.jpg output.jpg '))
    parser.add_argument('input_image', action='store', type=str, help='The input image as the full path, also excluding the file extension.')
    parser.add_argument('output_image', action='store', type=str, help='The filename (full path including file extension) of the `.png .jpg` file.')	
    args = parser.parse_args()

    main(args)