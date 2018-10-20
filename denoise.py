from __future__ import division

import cv2
import numpy as np
import sys
import os
import shutil

from denoise_fast import denoise

def denoiseVideo(filePath):
	
	outputPath = "%s.denoise.mp4" % (filePath)
	
	tempDir = "/tmp/denoise/"
	if os.path.exists(tempDir):
		shutil.rmtree(tempDir)
	os.makedirs(tempDir)
	
	# read in the video
	captureVideo = cv2.VideoCapture(filePath)
	fps = captureVideo.get(cv2.CAP_PROP_FPS)	
	
	# grab a rolling window of 5 frames
	frames = None
	nFrames = 0
	
	nOutputFrames = 0
	
	
	
	while(True):
		didGetImage,colorImage = captureVideo.read()
		if didGetImage == False:
			break
		
		#outputFrame = denoise(colorImage)
		
		outputFrame = cv2.fastNlMeansDenoisingColored(colorImage)
		
		nOutputFrames += 1
		print(".... %d" % (nOutputFrames))
		cv2.imwrite("%s/frame_%03d.png" % (tempDir, nOutputFrames), outputFrame)
		
		'''
		if frames is None:
			frames = np.zeros((5, colorImage.shape[0], colorImage.shape[1], colorImage.shape[2]), dtype="uint8")
		
		# first check if there is space in our window
		if nFrames < 5:
			np.copyto(frames[nFrames], colorImage)
			nFrames += 1
		
		if nFrames == 5:	
			# our rolling window is full; process the first middle frame, then roll the window, and add new frame at the end
			outputFrame = cv2.fastNlMeansDenoisingColoredMulti(frames, 2, 5)
						
			nOutputFrames += 1
			print(".... %d" % (nOutputFrames))
			cv2.imwrite("%s/frame_%03d.png" % (tempDir, nOutputFrames), outputFrame)
					
			# roll window and add new image
			for i in range(0,4):
				np.copyto(frames[i], frames[i+1])
			np.copyto(frames[4], colorImage)
		'''
	
	
	os.system("ffmpeg -framerate %f -i '/tmp/denoise/frame_%%03d.png' -c:v libx264 %s" % (fps, outputPath))

if __name__ == '__main__':
	
	if len(sys.argv) >= 2:
		filepath = sys.argv[1]
		denoiseVideo(filepath)
	else:
		print("usage: python denoise.py <path to video file>")