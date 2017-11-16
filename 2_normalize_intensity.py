import SimpleITK as sitk
import os
import numpy as np
import socket
from glob import glob


host = socket.gethostname()
if host == "cibl-thinkpad":
  baseDir = os.path.normpath('/home/avnishks/Desktop/DataSet/2_Data_NORM_nrrd')
  destDir = os.path.normpath('/home/avnishks/Desktop/DataSet/2_Data_NORM_nrrd/val/IntNormalized')
  numCores = 2
elif host == "R2Q5":
  baseDir = os.path.normpath('/data/Data_DeepLung_1/2_Data_NORM_nrrd/NSCLC_RT_DnSmpl_128')
  destDir = os.path.normpath('/data/Data_DeepLung_1/2_Data_NORM_nrrd/NSCLC_RT_DnSmpl_128/IntNormalized')
  numCores = 4
else:
  print('Unknown host!')
  sys.exit()
if not os.path.exists(destDir): os.mkdir(destDir)

# read nrrd image
files = glob(baseDir+'/*_img.nrrd')
print(files)

nrrdReader = sitk.ImageFileReader()
nrrdWriter = sitk.ImageFileWriter()
for x in files:
	nrrdReader.SetFileName(x)
	original_img_sitk = nrrdReader.Execute()
	img_cube = sitk.GetArrayFromImage(original_img_sitk)

	img_mean = np.mean(img_cube)
	img_std = np.std(img_cube)
	img_cube_norm = (img_cube - img_mean)/img_std

	# # write nrrd image
	imgSitk = sitk.GetImageFromArray(img_cube_norm)
	imgSitk.CopyInformation(original_img_sitk)

	outputFile = os.path.join(destDir,os.path.basename(nrrdReader.GetFileName()))
	nrrdWriter.SetFileName(outputFile)
	nrrdWriter.SetUseCompression(True)
	nrrdWriter.Execute(imgSitk)
