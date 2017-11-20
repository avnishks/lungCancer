import sys, os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from skimage import io, filters, measure, color
from skimage.morphology	import opening, closing, dilation, ball
from scipy import ndimage as ndi
from glob import glob
from functools import partial
from multiprocessing import Pool, Manager, Process
import multiprocessing
import socket

import exclude # make sure exclude.py is in the same folder
excludeList = exclude.getExcludeFiles('NSCLC_RT')

host = socket.gethostname()
if host == "cibl-thinkpad":
  baseDir = os.path.normpath('/home/avnishks/dfci/segmentation/Images')
  numCores = 4
elif host == "R2Q5":
  baseDir = os.path.normpath('/data/Data_DeepLung_1/2_Data_NORM_nrrd')
  numCores = 4
else:
  print('Unknown host!')
  sys.exit()

dataInput = os.path.join(baseDir, 'NSCLC_RT')
imageOutput = os.path.join(baseDir, 'NSCLC_RT/mask')
if not os.path.exists(imageOutput): os.mkdir(imageOutput)

nrrdReader = sitk.ImageFileReader()
nrrdWriter = sitk.ImageFileWriter()


def find_mask(imgCube):	
	# 1)thresholding
	imgGray = ((np.clip(imgCube, -1024.0, 3072.0))+1024)/4096*255
	threshold = filters.threshold_otsu(imgGray)
	mask = imgGray > threshold
	
	# 2)morphological denoising
	mask1 = closing(mask, ball(5))
	mask2 = dilation(mask1)
	maskClean = opening(mask2, ball(5))
	# fig1, ax1 = plt.subplots()
	# ax1.imshow(opened[128,:,:])
	# plt.show()

	# 3)connected regional analysis
	maskFill = ndi.binary_fill_holes(maskClean, structure=np.ones((3,3,3)))
	
	# largest volume -> body mask
	labelCube, numLabels = ndi.measurements.label(maskFill, structure=np.ones((3,3,3)))
	vol={}
	maxLabel = -1
	maxVolume = 0
	for labelNr in range(1, numLabels+1):
		vol[labelNr] = np.sum(labelCube[labelCube==labelNr])

	maxLabel = max(vol, key=vol.get)
	# print(maxLabel, vol[maxLabel])
	bodyMask = np.zeros(maskFill.shape)
	bodyMask[labelCube==maxLabel] = 1

	return bodyMask


def saveNrrd(patientID, mskSitk):
    mskFile = os.path.join(imageOutput, patientID+'_lun.nrrd')
    
    nrrdWriter.SetFileName(mskFile)
    nrrdWriter.SetUseCompression(True)
    nrrdWriter.Execute(mskSitk)


def runCore(patients, patientID):
	# imgFile = patients[patientID][0]
	mskFile = patients[patientID][0]

	print 'Process patient', patientID

	nrrdReader.SetFileName(mskFile)
	original_mskSitk = nrrdReader.Execute()
	imgCube = sitk.GetArrayFromImage(original_mskSitk)
	
	bodyMask = find_mask(imgCube)
	
	mskSitk = sitk.GetImageFromArray(bodyMask) #(fill_lungs.astype(int))
	mskSitk.CopyInformation(original_mskSitk)
	saveNrrd(patientID, mskSitk)


if __name__== "__main__":
	print "Read directories and searching for patients"
	files = glob(dataInput + '/*_img.nrrd')
	files = [x for x in files if not any(y in x for y in excludeList)]

	patients = {}
	for file in files:
		if 'img' in file:
			patientID = os.path.splitext(os.path.basename(file))[0].split('_')[0]
			# print(patientID)
			imgFile = file
			# mskFile = imgFile.replace('img', 'lun')
			# patients[patientID] = [imgFile, mskFile]
			patients[patientID] = [imgFile]
	print "Found ", str(len(patients)), " patients"
	# print(patients)
	
	with Manager() as manager:
		pool = multiprocessing.Pool(processes=numCores)
		pool.map(partial(runCore, patients), patients)
		pool.close()
		pool.join()
