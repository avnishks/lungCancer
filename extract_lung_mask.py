import sys, os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from skimage import io, filters, measure, color, segmentation
from skimage.morphology	import opening, closing, dilation, ball, cube, watershed, erosion
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_dilation, binary_closing 
from glob import glob
from functools import partial
from multiprocessing import Pool, Manager, Process
import multiprocessing
import socket

host = socket.gethostname()
if host == "cibl-thinkpad":
  baseDir = os.path.normpath('/home/avnishks/dfci/segmentation/Images')
  numCores = 2
elif host == "R2Q5":
  # baseDir = os.path.normpath('/data/Data_DeepLung_1/2_Data_NORM_nrrd')
  baseDir = os.path.normpath('/home/gpux/deepmedic/test')
  numCores = 4
else:
  print('Unknown host!')
  sys.exit()

dataInput = os.path.join(baseDir, 'NSCLC_RT')
imageOutput = os.path.join(baseDir, 'NSCLC_RT')
if not os.path.exists(imageOutput): os.mkdir(imageOutput)

nrrdReader = sitk.ImageFileReader()
nrrdWriter = sitk.ImageFileWriter()


# def find_mask(imgCube):	
# 	# 1)thresholding
# 	imgGray = ((np.clip(imgCube, -1024.0, 3072.0))+1024)/4096*255
# 	threshold = filters.threshold_otsu(imgGray)
# 	mask = imgGray > threshold
	
# 	# 2)morphological denoising
# 	mask1 = closing(mask, ball(5))
# 	mask2 = dilation(mask1)
# 	maskClean = opening(mask2, ball(5))
# 	# fig1, ax1 = plt.subplots()
# 	# ax1.imshow(opened[128,:,:])
# 	# plt.show()

# 	# 3)connected regional analysis
# 	maskFill = ndi.binary_fill_holes(maskClean, structure=np.ones((3,3,3)))
	
# 	# largest volume -> body mask
# 	labelCube, numLabels = ndi.measurements.label(maskFill, structure=np.ones((3,3,3)))
# 	vol={}
# 	maxLabel = -1
# 	maxVolume = 0
# 	for labelNr in range(1, numLabels+1):
# 		vol[labelNr] = np.sum(labelCube[labelCube==labelNr])

# 	maxLabel = max(vol, key=vol.get)
# 	# print(maxLabel, vol[maxLabel])
# 	bodyMask = np.zeros(maskFill.shape)
# 	bodyMask[labelCube==maxLabel] = 1

# 	return bodyMask


def generate_markers(image):
	# create internal markers
	marker_internal = image < -400
	marker_internal = segmentation.clear_border(marker_internal)
	marker_internal_labels = measure.label(marker_internal)
	areas = [r.area for r in measure.regionprops(marker_internal_labels)]
	areas.sort()
	if len(areas) > 2:
		for region in measure.regionprops(marker_internal_labels):
			if region.area < areas[-2]:
				for coordinates in region.coords:
					marker_internal_labels[coordinates[0], coordinates[1]]=0
	marker_internal = marker_internal_labels > 0
	
	# create external markers
	external_a = binary_dilation(marker_internal, iterations=15)
	external_b = binary_dilation(marker_internal, iterations=35)
	marker_external = external_b ^ external_a
	
	# create watershed marker matrix
	marker_watershed = np.zeros((512,512,512), dtype=np.int) #(512,512)
	marker_watershed += marker_internal * 255 
	marker_watershed += marker_external * 128
	
	
	return marker_internal, marker_external, marker_watershed

# def plot_2D_slices(imageVol, title,i=0)
# 	fig = plt.figure(int(i))
# 	plt.title(imageVol)
# 	plt.imshow(imgVol[265,:,:], cmap='gray')
# 	plt.savefig(fig)

def find_lung_mask(image):
	# find markers
	marker_internal, marker_external, marker_watershed = generate_markers(image)
	print ("marker_internal shape= ", np.shape(marker_internal))

	# create sobel-gradient
	# sobel_filtered_dx = ndi.sobel(image, 1)
	# sobel_filtered_dy = ndi.sobel(image, 0)
	# sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy) 
	sobel_gradient = ndi.generic_gradient_magnitude(image, ndi.sobel)
	# sobel_gradient *= 255.0 / np.max(sobel_gradient)
	print ("sobel_gradient shape= ", np.shape(sobel_gradient))
	
	# Watershed algorithm [slow step]
	watershedCube = watershed(sobel_gradient, marker_watershed)
	print ("watershedCube shape= ", np.shape(watershedCube))
	mskCube = erosion(watershedCube, cube(5))
	mskCube = opening(watershedCube, ball(6))

	mskCube = remove_small_objects(mskCube, 64)
	
	#reduce the image created by watershed algo to its outline
	outline1 = ndi.morphological_gradient(watershedCube, size=(3,3,3))
	print ("outline shape= ", np.shape(outline1))
	outline = outline1.astype(bool)
	
	#use internal marker and outline to generate the lung filter
	lungfilter = np.bitwise_or(marker_internal, outline)
	print ("lungfilter shape= ", np.shape(lungfilter))

	#Close holes in the lungfilter
	#fill_holes is not used here, since in some slices the heart would be reincluded by accident
	lungfilter = binary_closing(lungfilter, structure=np.ones((5,5,5)), iterations=3)
	print ("lungfilter after binary_closing shape= ", np.shape(lungfilter))
	
	# apply lungfilter (note the filtered area being assigned -2000 HU)
	segmented = np.where(lungfilter == 1, image, -2000*np.ones((512,512,512)))
	print ("segmented shape= ", np.shape(segmented))

	return segmented, lungfilter, outline1, watershedCube, sobel_gradient, marker_internal, marker_external, marker_watershed


def saveNrrd(patientID, mskSitk, **keyword):
    mskFile = os.path.join(imageOutput, patientID + keyword['name'] + '_lunExtract.nrrd')
    
    nrrdWriter.SetFileName(mskFile)
    nrrdWriter.SetUseCompression(True)
    nrrdWriter.Execute(mskSitk)


def runCore(patients, patientID):
	imgFile = patients[patientID][0]
	body_mskFile = patients[patientID][1]

	print 'Process patient', patientID

	nrrdReader.SetFileName(body_mskFile)
	original_body_mskSitk = nrrdReader.Execute()
	body_mskCube = sitk.GetArrayFromImage(original_body_mskSitk)
	body_mskCube = erosion(body_mskCube, ball(5))
	print ("body_mskCube after erosion")
	body_mskCubeSitk = sitk.GetImageFromArray(body_mskCube)
	body_mskCubeSitk.CopyInformation(original_body_mskSitk)
	saveNrrd(patientID, body_mskCubeSitk, name="_body_mskCube")

	nrrdReader.SetFileName(imgFile)
	original_imgSitk = nrrdReader.Execute()
	imgCube = sitk.GetArrayFromImage(original_imgSitk)

	# lungImg*bodyMsk + watershed --> lungMsk
	imgCube_clean = np.multiply(imgCube,body_mskCube)# + 1000*(1-body_mskCube)
	## do OTSU on imgCube_clean to remove the remaining table elements
	print ("imgCube_clean")
	imgCube_cleanSitk = sitk.GetImageFromArray(imgCube_clean)
	imgCube_cleanSitk.CopyInformation(original_body_mskSitk)
	saveNrrd(patientID, imgCube_cleanSitk, name="_imgCube_clean")

	#--------------- some test code ---------------#
	# Show some example markers from the middle        
	#test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(imgCube_clean[265])
	# print ("Internal Marker")
	# plt.imshow(test_patient_internal, cmap='gray')
	# plt.show()
	# print ("External Marker")
	# plt.imshow(test_patient_external, cmap='gray')
	# plt.show()
	# print ("Watershed Marker")
	# plt.imshow(test_patient_watershed, cmap='gray')
	# plt.show()
	#----------------------------------------------#

	#--------------- some test code ---------------#
	test_lungMask, test_lungfilter, test_outline, test_watershed,test_sobel_gradient, test_marker_internal, test_marker_external, test_marker_watershed = find_lung_mask(imgCube_clean)
	# 3)connected regional analysis
	# maskFill = ndi.binary_fill_holes(test_lungMask, structure=np.ones((3,3,3)))
	# maskFill = test_lungMask + 1800
	# maskFill[maskFill > 0] = 1
	# maskFill[maskFill <= 0] = 0
	# # largest volume -> lung mask, NO! Better solution: remove_small_objects(mskCube, 128)
	# labelCube, numLabels = ndi.measurements.label(maskFill, structure=np.ones((3,3,3)))
	# print(numLabels)
	# vol={}
	# maxLabel = -1
	# maxVolume = 0
	# for labelNr in range(0, numLabels+1):
	# 	vol[labelNr] = np.sum(labelCube[labelCube==labelNr])

	# maxLabel = max(vol, key=vol.get)
	# test_lungMask = np.zeros(maskFill.shape)
	# test_lungMask[labelCube==maxLabel] = 1
	# print ("test_lungMask final")

	print ("watershedCube")
	watershedCubeSitk = sitk.GetImageFromArray(test_watershed)
	watershedCubeSitk.CopyInformation(original_body_mskSitk)
	saveNrrd(patientID, watershedCubeSitk, name="_watershedCube")
	
	print ("outline")
	outlineSitk = sitk.GetImageFromArray(test_outline)
	outlineSitk.CopyInformation(original_body_mskSitk)
	saveNrrd(patientID, outlineSitk, name="outline")

	# print ("lungfilter")
	# lungfilterSitk = sitk.GetImageFromArray(lungfilter)
	# lungfilterSitk.CopyInformation(original_body_mskSitk)
	# saveNrrd(patientID, lungfilterSitk, name="lungfilter")

	print ("lungfilter after binary_closing")
	lungfilterSitk = sitk.GetImageFromArray(test_lungfilter.astype(int))
	lungfilterSitk.CopyInformation(original_body_mskSitk)
	saveNrrd(patientID, lungfilterSitk, name="_lungfilterBinaryClosing")
	# print ("Sobel Gradient")
	# plt.imshow(test_sobel_gradient, cmap='gray')
	# plt.show()
	# print ("Watershed Image")
	# plt.imshow(test_watershed, cmap='gray')
	# plt.show()
	# print ("Outline after reinclusion")
	# plt.imshow(test_outline, cmap='gray')
	# plt.show()
	# print ("Lungfilter after closing")
	# plt.imshow(test_lungfilter, cmap='gray')
	# plt.show()
	# print ("Segmented Lung")
	# plt.imshow(test_lungMask, cmap='gray')
	# plt.show()
	#----------------------------------------------#
	print ("segmented")
	lung_mskSitk = sitk.GetImageFromArray(test_lungMask)
	lung_mskSitk.CopyInformation(original_body_mskSitk)
	saveNrrd(patientID, lung_mskSitk, name="")
	

if __name__== "__main__":
	print "Read directories and searching for patients"
	files = glob(dataInput + '/*_img.nrrd')

	patients = {}
	for file in files:
		if 'img' in file:
			patientID = os.path.splitext(os.path.basename(file))[0].split('_')[0]
			# print(patientID)
			imgFile = file
			body_mskFile = imgFile.replace('img', 'lun')
			patients[patientID] = [imgFile, body_mskFile]
			# patients[patientID] = [imgFile]
	print "Found ", str(len(patients)), " patients"
	# print(patients)
	
	with Manager() as manager:
		pool = multiprocessing.Pool(processes=numCores)
		pool.map(partial(runCore, patients), patients)
		pool.close()
		pool.join()
