# list of patient without Lung volume+tumor mask+lung mask files
import os
from glob import glob
import numpy as np
import re

baseDir = os.path.normpath("/data/Data_DeepLung_1/2_Data_NORM_nrrd/NSCLC_RT")
fileNames_img = glob(baseDir+"/*_img.nrrd") #lung image
fileNames_tum = glob(baseDir+"/*_tum.nrrd") #tumor mask
fileNames_lun = glob(baseDir+"/*_lun.nrrd") #lung mask

# patientID = os.path.splitext(os.path.basename(file))[0].split('_')[0]
regex = re.compile(r'\d+')
patientID_img = [regex.findall(x)[-1] for x in fileNames_img]
patientID_tum = [regex.findall(x)[-1] for x in fileNames_tum]
patientID_lun = [regex.findall(x)[-1] for x in fileNames_lun]
#print(patientID_img)

excludeList = list((set(patientID_img) | set(patientID_tum)) - (set(patientID_img) & set(patientID_tum)))
print(excludeList)
