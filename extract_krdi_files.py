import os
import shutil
from os import listdir
from os.path import isfile, join
import os
import shutil
from os import listdir
from os.path import isfile, join

from Configs import SharedConfigurations

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

###################################################################################################

def checkIfDuplicates(listOfElems):
    ''' Check if given list contains any duplicates '''
    cnt=0
    duplicates=[]
    new_ids=[]
    for elem in listOfElems:
        if listOfElems.count(elem) > 1:
            duplicates.append(elem)
            cnt+=1
        else :
            new_ids.append(elem)
    print(cnt)
    return new_ids


# add ids in list :
def add_ids(path):
    krdis = os.listdir(path)
    ids=[]
    for krdi in krdis:
        id_krdi = krdi.split(".")[0]
        id_krdi = id_krdi.split("_")[1]
        ids.append(id_krdi)
    return ids

def copy_filtered_krdi(path1,path2,new_ids):
    krdis = os.listdir(path1)
    for krdi in krdis:
        id_krdi = krdi.split(".")[0]
        id_krdi = id_krdi.split("_")[1]
        if id_krdi in new_ids:
            original = path1 + "\\" + krdi
            target = path2 + "\\" + krdi
            shutil.copyfile(original,target)

############################################################################################################
conf=SharedConfigurations()
path = conf.sirius_folder
raw_krdis=conf.raw_krdis_folder
# 1. STEP extracting ***ALL**** .krdi files from WUERTG_IRIIS_DISK_COPY/system_OLD
#extract_all_krdi_files(path,new_path)

# 2. STEP make new folder with unique id-s (no duplicates!)
raw_krdis_unique=conf.raw_krdis_unique

ids=add_ids(raw_krdis)
filtered_ids=checkIfDuplicates(ids)
copy_filtered_krdi(raw_krdis,raw_krdis_unique,filtered_ids)

# 3. STEP use KRDI converter to convert .krdi files  (setting : don't delete .krdi file while converting)
# After that, continue with script "prepare_dataset.py"
