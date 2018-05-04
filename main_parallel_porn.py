#export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH
import sys
import cv2
import math
import numpy as np
import struct
import pickle
import colorsys


from joblib import Parallel, delayed

def read_videolist(fname, videolist):

	with open(fname, 'r') as f:
		for line in f:
			videolist.append(line)
	return;


def creating_tensor_series(features, tensor_series, bins):

	b = bins
	matrix = np.zeros((b, b), dtype=np.float)
	matrix2 = np.zeros((b, b), dtype=np.float)

	for f in range(0,len(features)):

		for i in range(0,b):
			for j in range(0,b):
				matrix[i][j] += features[f][i] * features[f][j]

		for i in range(0,b):
			for j in range(0,b):
				matrix2[i][j] += matrix[i][j]

		#normalizing l2 - each image has a tensor
		#mean = 0.0
		#for i in range(0, b):
		#	for j in range(0, b):
		#		mean += matrix2[i][j] * matrix2[i][j]


		#for i in range(0,b):
		#	for j in range(0,b):
		#		matrix2[i][j] /= math.sqrt(mean)

		tensor_series.append(matrix2)

	return tensor_series

def creating_final_tensor(tensor_series, bins):
	b = bins
        final_tensor = np.zeros((b, b), dtype=np.float)

	for f in range(0, len(tensor_series)):
		for i in range(0,b):
			for j in range(0,b):
				final_tensor[i][j] += tensor_series[f][i][j]
	

	#normalizing l2
	mean = 0.0
	for i in range(0, b):
		for j in range(0, b):
			mean += final_tensor[i][j] * final_tensor[i][j]
	
	if mean > 0.00000:
		for i in range(0,b):
			for j in range(0,b):
				final_tensor[i][j] /= math.sqrt(mean)

	return final_tensor


def extracting_hog(frame, bins):
	winSize = (64,64)
	blockSize = (16,16)
	blockStride = (8,8)
	#cellSize = (8,8)
	cellSize = (16,16)
	nbins = bins
	derivAperture = 1
	winSigma = 4.
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 0
	nlevels = 64
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
	#compute(img[, winStride[, padding[, locations]]]) -> descriptors
	winStride = (8,8)
	padding = (8,8)
	locations = ((10,20),)
	hist = hog.compute(frame,winStride,padding,locations)

	return hist

def process(video, tensor_series, bins):

	pos = video.index('\n')
    	vid = cv2.VideoCapture(video[0:pos])

	while(vid.isOpened()):
		#Capture frame-by-frame
		ret, frame = vid.read()
		if ret == False:
			break
	
		#extracting hog
		features = []
		features.append(extracting_hog(frame, bins))

		#creating_tensor_series for frame
		tensor_series = creating_tensor_series(features, tensor_series, bins) 	

	vid.release()
	
	#accumulate temporal information - for each video
	final_tensor = creating_final_tensor(tensor_series, bins)
	
	print final_tensor

	name = video[0:pos]
	name = name.split('/')
    	name_file = 'tensors/tensor_from' + name[1] +'hog' + str(bins) + '.pkl'

    	with open(name_file, 'wb') as file:
		pickle.dump({'tensor_series': final_tensor}, file)


#read videolist
#read number of bins

videos = str(sys.argv[1])
bins = int(sys.argv[2])

videolist = []
read_videolist(videos, videolist)

tensor_series = []

#extract feature and create tensor
Parallel(n_jobs=4)(delayed(process)(videolist[i], tensor_series, bins) for i in range(len(videolist)))
#Parallel(n_jobs=4)(delayed(process)(videolist[i], tensor_series, bins) for i in range(4))

