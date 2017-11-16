import sys, glob, ntpath, csv, pickle
import os, fnmatch, shutil, subprocess
import multiprocessing, socket
from multiprocessing import Process, Manager
from shutil import copyfile
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy.ndimage as ndimage
from scipy.ndimage import measurements
from scipy import interpolate

from glob import glob
from skimage import measure
from IPython.utils import io

import exclude # make sure exclude.py is in the same folder
excludeList = exclude.getExcludeFiles('NSCLC_RT')

newSize = 128
cropedSize = 128 # Size to be croped after downsampling to get correct sizes and spacings
newSpacing = None #3.0

PNG = False
DataSet = str(1) # 1 or 2 for Dataset

nrrdReader = sitk.ImageFileReader()
nrrdWriter = sitk.ImageFileWriter()

numCores = None # Will be set for each host
baseDir = None
host = socket.gethostname()
if host == "cibl-thinkpad":
  baseDir = os.path.normpath('/home/avnishks/Desktop/DataSet/2_Data_NORM_nrrd')
  numCores = 2
elif host == "R2Q5":
  baseDir = os.path.normpath('/data/Data_DeepLung_1/2_Data_NORM_nrrd')
  numCores = 4
else:
  print('Unknown host!')
  sys.exit()

# Set path and file names
dataInput = os.path.join(baseDir, 'NSCLC_RT')
imageOutput = os.path.join(baseDir, 'NSCLC_RT_DnSmpl_128')
# imageOutput_PNG = os.path.join(baseDir, 'NSCLC_RT_DnSmpl_png')
pklOutput = os.path.join(baseDir, '3_networkData') #??

if not os.path.exists(imageOutput): os.mkdir(imageOutput)
# if not os.path.exists(imageOutput_PNG): os.mkdir(imageOutput_PNG)
if not os.path.exists(pklOutput): os.mkdir(pklOutput)


def resampleSitk(imgSitk, mskSitk, tum_mskSitk):
  global newSpacing
  global newSize

  origSpacing = imgSitk.GetSpacing()
  origSize = imgSitk.GetSize()
  
  if newSpacing == None:
    newSpacing = origSize[0]*origSpacing[0]/newSize
  else:
    newSize = int(origSize[0]*origSpacing[0]/newSpacing)

  # resampledSize = [int(origSize[0]*origSpacing[0]/newSpacing),
  #                 int(origSize[1]*origSpacing[1]/newSpacing),
  #                 int(origSize[2]*origSpacing[2]/newSpacing)]

  resFilter = sitk.ResampleImageFilter()
  imgSitk = resFilter.Execute(imgSitk,
                              [newSize,newSize,newSize],
                              sitk.Transform(),
                              sitk.sitkLinear,
                              imgSitk.GetOrigin(),
                              [newSpacing,newSpacing,newSpacing],
                              imgSitk.GetDirection(),
                              0,
                              imgSitk.GetPixelIDValue())
  
  resFilter = sitk.ResampleImageFilter()
  mskSitk = resFilter.Execute(mskSitk,
                              [newSize,newSize,newSize],
                              sitk.Transform(),
                              sitk.sitkNearestNeighbor,
                              imgSitk.GetOrigin(),
                              [newSpacing,newSpacing,newSpacing],
                              imgSitk.GetDirection(),
                              0,
                              mskSitk.GetPixelIDValue())

  resFilter = sitk.ResampleImageFilter()
  tum_mskSitk = resFilter.Execute(tum_mskSitk,
                              [newSize,newSize,newSize],
                              sitk.Transform(),
                              sitk.sitkNearestNeighbor,
                              imgSitk.GetOrigin(),
                              [newSpacing,newSpacing,newSpacing],
                              imgSitk.GetDirection(),
                              0,
                              tum_mskSitk.GetPixelIDValue())
  
  return imgSitk, mskSitk, tum_mskSitk, origSize, origSpacing


def cropSitk(dataSitk, air):
  # 1) Crop down bigger axes
  oldSize = dataSitk.GetSize()
  sizeDif = [cropedSize - oldSize[0], cropedSize - oldSize[1], cropedSize - oldSize[2]]
  # print oldSize

  newSizeDown = [max(0, int((oldSize[0] - cropedSize) / 2)),
                 max(0, int((oldSize[1] - cropedSize) / 2)),
                 max(0, int((oldSize[2] - cropedSize) / 2))]
  newSizeUp = [max(0, oldSize[0] - cropedSize - newSizeDown[0]),
               max(0, oldSize[1] - cropedSize - newSizeDown[1]),
               max(0, oldSize[2] - cropedSize - newSizeDown[2])]

  # print newSizeDown, newSizeUp

  cropFilter = sitk.CropImageFilter()
  cropFilter.SetUpperBoundaryCropSize(newSizeUp)
  cropFilter.SetLowerBoundaryCropSize(newSizeDown)
  dataSitk = cropFilter.Execute(dataSitk)

  # 2) Pad smaller axes
  oldSize = dataSitk.GetSize()
  # print oldSize

  newSizeDown = [max(0, int((cropedSize - oldSize[0]) / 2)),
                 max(0, int((cropedSize - oldSize[1]) / 2)),
                 max(0, int((cropedSize - oldSize[2]) / 2))]
  newSizeUp = [max(0, cropedSize - oldSize[0] - newSizeDown[0]),
               max(0, cropedSize - oldSize[1] - newSizeDown[1]),
               max(0, cropedSize - oldSize[2] - newSizeDown[2])]

  # print newSizeDown, newSizeUp

  padFilter = sitk.ConstantPadImageFilter()
  padFilter.SetConstant(air)
  padFilter.SetPadUpperBound(newSizeUp)
  padFilter.SetPadLowerBound(newSizeDown)
  dataSitk = padFilter.Execute(dataSitk)

  finalSize = dataSitk.GetSize()
  finalSpacing = dataSitk.GetSpacing()

  return dataSitk, finalSize, finalSpacing, sizeDif





# def expandSitk(imgSitk, mskSitk):
#   oldSize = imgSitk.GetSize()
#   finalSize = cropedSize
  
#   sizeDif = [finalSize-oldSize[0], finalSize-oldSize[1], finalSize-oldSize[2]]
  
#   newSizeDown = [max(0, int(sizeDif[0]/2)),
#                  max(0, int(sizeDif[1]/2)),
#                  max(0, int(sizeDif[2]/2))]
#   newSizeUp = [max(0, sizeDif[0]-newSizeDown[0]),
#                max(0, sizeDif[1]-newSizeDown[1]),
#                max(0, sizeDif[2]-newSizeDown[2])]
  
#   padFilter = sitk.ConstantPadImageFilter()
#   padFilter.SetPadUpperBound(newSizeUp)
#   padFilter.SetPadLowerBound(newSizeDown)
  
#   padFilter.SetConstant(-1024)
#   imgSitk = padFilter.Execute(imgSitk)
  
#   padFilter.SetConstant(0)
#   mskSitk = padFilter.Execute(mskSitk)
  
#   finalSize = imgSitk.GetSize()
#   finalSpacing = imgSitk.GetSpacing()

#   print finalSize, finalSpacing, sizeDif
  
#   return imgSitk, mskSitk, finalSize, finalSpacing, sizeDif

def savePNG(patientID, imgSitk, mskSitk): #modify to add tum_msk
  
  imgCube = sitk.GetArrayFromImage(imgSitk)
  mskCube = sitk.GetArrayFromImage(mskSitk)
  mskCube[mskCube!=1] = 0
  
  maskIndices = np.where(mskCube != 0)
  if len(maskIndices[0]) == 0:
    print 'ERROR - Found empty mask for patient', patientID
    return

  BB = [np.min(maskIndices[0]), np.max(maskIndices[0]),
        np.min(maskIndices[1]), np.max(maskIndices[1]),
        np.min(maskIndices[2]), np.max(maskIndices[2])]
  center = [(BB[1]-BB[0])/2+BB[0],
            (BB[3]-BB[2])/2+BB[2],
            (BB[5]-BB[4])/2+BB[4]]
  
  fig,ax = plt.subplots(1, 3, figsize=(32,16))
  
  ax[0].imshow(imgCube[center[0],:,:], cmap='gray')
  ax[1].imshow(imgCube[:,center[1],:], cmap='gray')
  ax[2].imshow(imgCube[:,:,center[2]], cmap='gray')
  
  ax[0].imshow(mskCube[center[0],:,:], cmap='jet', alpha=0.4)
  ax[1].imshow(mskCube[:,center[1],:], cmap='jet', alpha=0.4)
  ax[2].imshow(mskCube[:,:,center[2]], cmap='jet', alpha=0.4)
  
  fileName = os.path.join(imageOutput_PNG, (str(patientID)+".png"))
  plt.savefig(fileName)
  plt.close(fig)


def saveNrrd(patientID, imgSitk, mskSitk, tum_mskSitk):
    imgFile = os.path.join(imageOutput, patientID+'_img.nrrd')
    mskFile = os.path.join(imageOutput, patientID+'_lun.nrrd')#'_msk.nrrd')
    tum_mskFile = os.path.join(imageOutput, patientID+'_tum.nrrd')
    
    nrrdWriter.SetFileName(imgFile)
    nrrdWriter.SetUseCompression(True)
    nrrdWriter.Execute(imgSitk)
    
    nrrdWriter.SetFileName(mskFile)
    nrrdWriter.SetUseCompression(True)
    nrrdWriter.Execute(mskSitk)

    nrrdWriter.SetFileName(tum_mskFile)
    nrrdWriter.SetUseCompression(True)
    nrrdWriter.Execute(tum_mskSitk)


def runCore(patients, coreNr, resultDict):
  
  for patientID in patients.keys():
    
    imgFile = patients[patientID][0]
    mskFile = patients[patientID][1]
    tum_mskFile = patients[patientID][2]

    print 'Process patient', patientID
    
    nrrdReader.SetFileName(imgFile)
    imgSitk = nrrdReader.Execute()
    
    nrrdReader.SetFileName(mskFile)
    mskSitk = nrrdReader.Execute()

    nrrdReader.SetFileName(tum_mskFile)
    tum_mskSitk = nrrdReader.Execute()
    
    imgSitk, mskSitk, tum_mskSitk, origSize, origSpacing = resampleSitk(imgSitk, mskSitk, tum_mskSitk)
    
    imgSitk, finalSize_I, finalSpacing_I, sizeDif_I = cropSitk(imgSitk, -1024)
    mskSitk, finalSize_M, finalSpacing_M, sizeDif_M = cropSitk(mskSitk, 0)
    tum_mskSitk, finalSize_TM, finalSpacing_TM, sizeDif_TM = cropSitk(tum_mskSitk, 0)

    #modify to add tum_msk
    if not (cropedSize, cropedSize, cropedSize) == finalSize_I == finalSize_M:
      print 'Wrong final size', patientID, finalSize_I, finalSize_M
    if not (newSpacing, newSpacing, newSpacing) == finalSpacing_M == finalSpacing_M:
      print 'Wrong final spacing', patientID, finalSpacing_I, finalSpacing_M

    saveNrrd(patientID, imgSitk, mskSitk, tum_mskSitk)
    
    if PNG:
      savePNG(patientID, imgSitk, mskSitk, tum_mskSitk)
    
    resultDict[patientID] = [imgFile, mskFile, tum_mskFile,
                             origSize, origSpacing,
                             finalSize_I, finalSpacing_I,
                             sizeDif_I]


if __name__== "__main__":
  print "Read directories and searching for patients"
  files = glob(dataInput + '/*')
  files = [x for x in files if not any(y in x for y in excludeList)]
  
  patients = {}
  for file in files:
    if 'img' in file:
      patientID = os.path.splitext(os.path.basename(file))[0].split('_')[0]
      imgFile = file
      mskFile = imgFile.replace('img', 'lun')
      tum_mskFile = imgFile.replace('img', 'tum')
      patients[patientID] = [imgFile, mskFile, tum_mskFile]
  print "Found ", str(len(patients)), " patients"

  with Manager() as manager:
    resultDict = manager.dict()
    processes = []
    for coreCount in range(numCores):
      part = {key:patients[key] for i, key in enumerate(patients) if i % numCores == coreCount}
      print 'Number of patients for core nr', coreCount, ':', len(part)
      p = Process(target=runCore, args=(part, coreCount, resultDict,))
      p.start()
      processes.append(p)
    for p in processes:
      p.join()
    
    resultDictNew = dict(resultDict)
    resultsFileName = os.path.join(imageOutput, 'results_112.pkl')
    with open(resultsFileName, 'wb') as resultsFile:
      pickle.dump(resultDictNew, resultsFile, pickle.HIGHEST_PROTOCOL)

    # Copy pkl file to data folder for training
    secondPickle = os.path.join(pklOutput, 'results_112.pkl')
    copyfile(resultsFileName, secondPickle)
