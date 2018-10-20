from __future__ import division

import cv2
import numpy as np
import sys
import os
import shutil

import multiprocessing
from multiprocessing import Process
from multiprocessing import Pool
import time

cv2.setNumThreads(1)

numWorkers = multiprocessing.cpu_count()

outputPath = None

tempDir = "/tmp/denoise/"

if False:
	if os.path.exists(tempDir):
		shutil.rmtree(tempDir)

if os.path.exists(tempDir) == False:
	os.makedirs(tempDir)

def pathForFrameNumber(frameNum):
	global tempDir
	return "%s/frame_%03d.png" % (tempDir, frameNum)

processOutputQueue = multiprocessing.Manager().Queue()
processInputQueue = multiprocessing.Manager().Queue(maxsize=numWorkers)
totalFramesFinalized = 0

def Process_DenoiseFrame():
	global processOutputQueue
	global processInputQueue

	while True:
		args = processInputQueue.get()
		frameNum = args[0]
		frame = args[1]
		outputFrame = cv2.fastNlMeansDenoisingColored(frame)
		cv2.imwrite(pathForFrameNumber(frameNum), outputFrame)
		print(".... %d" % (frameNum))
		processOutputQueue.put( (frameNum) )

def checkForResults(nFrames):
	global totalFramesFinalized
	global processOutputQueue
	global processInputQueue
	global tempDir
	
	# check to see if any frames have completed
	while True:
		try:
			processOutputQueue.get_nowait()
			totalFramesFinalized += 1
		except:
			break
	
	return (totalFramesFinalized == nFrames)
	
	

def denoiseVideo(filePath):
	global outputPath
	global processOutputQueue
	global processInputQueue
	global totalFramesFinalized
	
	outputPath = "%s.denoise.mp4" % (filePath)
	
	# read in the video
	captureVideo = cv2.VideoCapture(filePath)
	fps = captureVideo.get(cv2.CAP_PROP_FPS)
		
	for i in range(0, numWorkers):
		process = Process(target=Process_DenoiseFrame, args=())
		process.daemon = True
		process.start()
	
	nFrames = 0
	while True:
		didGetImage,colorImage = captureVideo.read()
		if didGetImage == False:
			break
		
		if os.path.exists(pathForFrameNumber(nFrames)) == False:
			# create a job for one of our workers to do
			print('submitting job for frame %d' % (nFrames))
			processInputQueue.put((nFrames, colorImage))
		else:
			print('skipping job for frame %d' % (nFrames))
			totalFramesFinalized += 1
		nFrames += 1
					
		checkForResults(None)
	
	# ensure we get all of the results...
	while checkForResults(nFrames) == False:
		pass
	
	os.system("ffmpeg -framerate %f -i '/tmp/denoise/frame_%%03d.png' -c:v libx264 %s" % (fps, outputPath))


if __name__ == '__main__':
	
	if len(sys.argv) >= 2:
		filepath = sys.argv[1]
		denoiseVideo(filepath)
	else:
		print("usage: python denoise.py <path to video file>")