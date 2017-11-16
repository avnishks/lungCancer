
# These are the patient IDs which are excluded for lung training

import numpy as np

excludeFiles = {}
excludeFiles['moffit'] = ['R0009','R0018','R0030','R0038','R0058','R0068','R0077','R0080','R0081','R0082','R0084','R0085',
          'R0091','R0102','R0122','R0147','R0162','R0163','R0169','R0172','R0186','R0203','R0206','R0215',
          'R0216','R0236','R0238','R0214','R0252','R0262','R0267','R0271']

excludeFiles['moffit_spore'] = ['325364','400515','543256','562116']


#most of these don't have both lung and tumor masks
excludeFiles['NSCLC_RT'] = ['0187','0200','0202','0210','0355','0625','0652','1030','1050','0178', '1305', '1306', '1307', '0259', '1309', '1744', '1159', '1321', '0691', '0976', '0959', '0956', '1146', '0955', '0875', '1312', '1739', '1738', '1240', '0839', '1733', '0916', '0897', '0896', '0205', '0145', '0892', '0147', '1757', '1756', '1755', '1754', '0677', '1752', '1316', '1315', '1314', '1313', '0108', '1214', '1155', '1137', '1238', '1239', '1319', '1318', '0840', '0710', '0843', '0903', '1745', '0940', '0984', '0944', '0904', '0949', '1020', '1728', '1729', '1726', '1740', '0923', '1741', '0157', '1748', '1244', '1320', '0574', '0661', '1746', '1747', '0664', '0665', '1743', '1010', '0044']#latest_1

def getExcludeFiles(dataset):
  if dataset in excludeFiles:
    return excludeFiles[dataset]
  else:
    return []

def getAllExcludeFiles():
  fileList = []
  for key in excludeFiles.keys():
    fileList = np.hstack([fileList, excludeFiles[key]])
  return fileList
