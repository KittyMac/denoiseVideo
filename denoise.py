from __future__ import division

import cv2
import numpy as np
import sys

def denoiseVideo(filePath):
	
	# read in the video
	captureVideo = cv2.VideoCapture(filePath)
	fps = captureVideo.get(cv2.CAP_PROP_FPS)
	
	# write out the new video (deferred till after we read the first image)
	writeVideo = None
	
	
	# grab a rolling window of 5 frames
	frames = None
	nFrames = 0
	
	nOutputFrames = 0
	
	while(True):
		didGetImage,colorImage = captureVideo.read()
		if didGetImage == False:
			break
		
		if writeVideo is None:
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			writeVideo = cv2.VideoWriter("%s.denoise.mp4" % (filePath), fourcc, fps, (colorImage.shape[0], colorImage.shape[1]))
		
		if frames is None:
			frames = np.zeros((5, colorImage.shape[0], colorImage.shape[1], colorImage.shape[2]), dtype="uint8")
		
		# first check if there is space in our window
		if nFrames < 5:
			np.copyto(frames[nFrames], colorImage)
			nFrames += 1
		
		if nFrames == 5:	
			# our rolling window is full; process the first middle frame, then roll the window, and add new frame at the end
			outputFrame = cv2.fastNlMeansDenoisingColoredMulti(frames, 2, 5)
			
			writeVideo.write(outputFrame)
			
			nOutputFrames += 1
			print(".... %d" % (nOutputFrames))
			#cv2.imwrite("/tmp/output%d.png" % (nOutputFrames), outputFrame)
		
			# roll window and add new image
			for i in range(0,4):
				np.copyto(frames[i], frames[i+1])
			np.copyto(frames[4], colorImage)
	
	writeVideo.release()


if __name__ == '__main__':
	
	if len(sys.argv) >= 2:
		filepath = sys.argv[1]
		denoiseVideo(filepath)
	else:
		print("usage: python denoise.py <path to video file>")