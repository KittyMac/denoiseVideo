from __future__ import division

import cv2
import numpy as np
import sys
import os
import shutil

tempDir = "/tmp/denoise/"
if os.path.exists(tempDir):
	if raw_input("Existing progress frames found, delete? (y/N): ") == "y":
		shutil.rmtree(tempDir)
if os.path.exists(tempDir) == False:
	os.makedirs(tempDir)

def pathForFrameNumber(frameNum):
	global tempDir
	return "%s/frame_%03d.png" % (tempDir, frameNum)

def denoiseVideo(filePath):
	
	outputPath = "%s.denoise.mp4" % (filePath)
		
	# read in the video
	captureVideo = cv2.VideoCapture(filePath)
	fps = captureVideo.get(cv2.CAP_PROP_FPS)	
	
	# grab a rolling window of 5 frames
	nOutputFrames = 0
	
	while(True):
		didGetImage,colorImage = captureVideo.read()
		if didGetImage == False:
			break
		
		if os.path.exists(pathForFrameNumber(nOutputFrames)) == False:
			print(".... %d" % (nOutputFrames))
			outputFrame = cv2.fastNlMeansDenoisingColored(colorImage)
			cv2.imwrite("%s/frame_%03d.png" % (tempDir, nOutputFrames), outputFrame)
		else:
			print('skipping frame %d' % (nOutputFrames))
		
		nOutputFrames += 1
		
	
	os.system("ffmpeg -framerate %f -i '/tmp/denoise/frame_%%03d.png' -c:v libx264 %s" % (fps, outputPath))

if __name__ == '__main__':
	
	if len(sys.argv) >= 2:
		filepath = sys.argv[1]
		denoiseVideo(filepath)
	else:
		print("usage: python denoise.py <path to video file>")