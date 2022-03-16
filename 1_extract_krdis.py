import os
import shutil
from os import listdir
from os.path import isfile, join
path = r"G:\Matea\WUERTH_IRIIS_DISK_COPY\system_OLD"
krdis= r"G:\Matea\krdis"
roots=os.listdir(path)
for root in roots:
	root_path=path+"\\"+root
	directories = os.listdir(root_path)
	for dir in directories:
		dir_path=root_path+"\\"+dir
		files=os.listdir(dir_path)
		for file_name in files:
			extension = os.path.splitext(file_name)[1]
			if(extension == ".krdi"):
				file_path= dir_path+"\\"+file_name
				print(file_path)
				original=file_path
				target=krdis+"\\"+str(file_name)
				shutil.copyfile(original, target)