import os, sys, socket
import shutil
from glob import glob


host = socket.gethostname()
if host == "R2Q5":
  baseDir = os.path.normpath('/data/Data_DeepLung_1/2_Data_NORM_nrrd')
  numCores = 4
else:
  print('Unknown host!')
  sys.exit()

# TRAINING CFG FILES
# trainSrcDir = baseDir + "/NSCLC_RT_DnSmpl_128/train" 
# trainDestDir = baseDir + "/NSCLC_RT_DnSmpl_128/train"

# for root, subFolders, files in os.walk(trainSrcDir):
# 	for file in files:
# 		subFolder = os.path.join(trainDestDir, file[:4])
# 		if not os.path.isdir(subFolder):
# 			os.makedirs(subFolder)
# 		shutil.move(os.path.join(root, file), subFolder)

# files_img = glob(trainDestDir + '/*/*_img.nii.gz')
# # files_lun = glob(trainDestDir + '/*/*_lun.nii.gz')
# files_tum = glob(trainDestDir + '/*/*_tum.nii.gz')

# outfile_img = open("/home/gpux/deepmedic/examples/configFiles/deepMedic/train/trainChannels_lung.cfg", "w")
# print >> outfile_img, "\n".join(x for x in files_img)
# outfile_img.close()

# # outfile_lun = open("/home/gpux/deepmedic/examples/configFiles/deepMedic/train/trainRoiMasks.cfg", "w")
# # print >> outfile_lun, "\n".join(x for x in files_lun)
# # outfile_lun.close()

# outfile_tum = open("/home/gpux/deepmedic/examples/configFiles/deepMedic/train/trainGtLabels.cfg", "w")
# print >> outfile_tum, "\n".join(x for x in files_tum)
# outfile_tum.close()


# # VALIDATION CFG FILES
# valSrcDir = baseDir + "/NSCLC_RT_DnSmpl_128/validation" 
# valDestDir = baseDir + "/NSCLC_RT_DnSmpl_128/validation"

# for root, subFolders, files in os.walk(valSrcDir):
# 	for file in files:
# 		subFolder = os.path.join(valDestDir, file[:4])
# 		if not os.path.isdir(subFolder):
# 			os.makedirs(subFolder)
# 		shutil.move(os.path.join(root, file), subFolder)

# files_img = glob(valDestDir + '/*/*_img.nii.gz')
# # files_lun = glob(valDestDir + '/*/*_lun.nii.gz')
# files_tum = glob(valDestDir + '/*/*_tum.nii.gz')
# files_names = glob(valDestDir + '/*/*_tum.nii.gz')

# outfile_img = open("/home/gpux/deepmedic/examples/configFiles/deepMedic/train/validation/validationChannels_lung.cfg", "w")
# print >> outfile_img, "\n".join(x for x in files_img)
# outfile_img.close()

# # outfile_lun = open("/home/gpux/deepmedic/examples/configFiles/deepMedic/train/validation/validationRoiMasks.cfg", "w")
# # print >> outfile_lun, "\n".join(x for x in files_lun)
# # outfile_lun.close()

# outfile_tum = open("/home/gpux/deepmedic/examples/configFiles/deepMedic/train/validation/validationGtLabels.cfg", "w")
# print >> outfile_tum, "\n".join(x for x in files_tum)
# outfile_tum.close()

# files_names_edit=[]
# for x in files_names:
# 	files_names_edit.append("pred_" + str(x[-15:-11]) + ".nii.gz")

# outfile_names = open("/home/gpux/deepmedic/examples/configFiles/deepMedic/train/validation/validationNamesOfPredictions.cfg", "w")
# print >> outfile_names, "\n".join([x for x in files_names_edit])
# outfile_names.close()


# TEST CFG FILES
testSrcDir = baseDir + "/NSCLC_RT_DnSmpl_128/test" 
testDestDir = baseDir + "/NSCLC_RT_DnSmpl_128/test"

for root, subFolders, files in os.walk(testSrcDir):
	for file in files:
		subFolder = os.path.join(testDestDir, file[:4])
		if not os.path.isdir(subFolder):
			os.makedirs(subFolder)
		shutil.move(os.path.join(root, file), subFolder)

files_img = glob(testDestDir + '/*/*_img.nii.gz')
# files_lun = glob(testDestDir + '/*/*_lun.nii.gz')
files_tum = glob(testDestDir + '/*/*_tum.nii.gz')
files_names = glob(testDestDir + '/*/*_tum.nii.gz')

outfile_img = open("/home/gpux/deepmedic/examples/configFiles/deepMedic/test/testChannels_lung.cfg", "w")
print >> outfile_img, "\n".join(x for x in files_img)
outfile_img.close()

# outfile_lun = open("/home/gpux/deepmedic/examples/configFiles/deepMedic/test/testRoiMasks.cfg", "w")
# print >> outfile_lun, "\n".join(x for x in files_lun)
# outfile_lun.close()

outfile_tum = open("/home/gpux/deepmedic/examples/configFiles/deepMedic/test/testGtLabels.cfg", "w")
print >> outfile_tum, "\n".join(x for x in files_tum)
outfile_tum.close()

files_names_edit=[]
for x in files_names:
	files_names_edit.append("pred_" + str(x[-15:-11]) + ".nii.gz")

outfile_names = open("/home/gpux/deepmedic/examples/configFiles/deepMedic/test/testNamesOfPredictions.cfg", "w")
print >> outfile_names, "\n".join([x for x in files_names_edit])
outfile_names.close()
