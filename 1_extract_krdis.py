import os
import shutil
from os import listdir
from os.path import isfile, join
from Configs_parsing import SharedConfigurations
# extracting ***ALL**** .krdi files from WUERTG_IRIIS_DISK_COPY/system_OLD that Paul gave me
def extract_all_krdi_files(old_path,new_path):
	path=old_path
	krdis=new_path
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

conf=SharedConfigurations()
path = conf.sirius_folder

krdis = r"G:\Matea\krdis"


#if __name__ == '__main__':
# extract_all_krdi_files(path,krdis)