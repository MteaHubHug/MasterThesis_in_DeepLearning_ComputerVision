import os
import shutil
from os import listdir
from os.path import isfile, join


path = r"G:\Matea\WUERTH_IRIIS_DISK_COPY\system_NEW"
krdi_path = r"G:\Matea\krdis4"
triplets_path=r"G:\Matea\triplets"
def get_ids(path):
    krdis = os.listdir(path)
    krdi_ids=[]
    for krdi in krdis:
        id_krdi= krdi.split("_")[0]
        #print(id_krdi)
        krdi_ids.append(id_krdi)
    return krdi_ids

#krdi_ids=get_ids(krdi_path)


def matching_mapping(krdi_ids,path,triplets_path):
    boxes=os.listdir(path)
    match=0
    for dir in boxes:
        file_path = path + "\\" + dir
        files=os.listdir(file_path)
        for file in files:
                id=file.split("_")[0]
                original = file_path + "\\" + file
                target = triplets_path + "\\" + str(file)
                if id in krdi_ids:
                    match+=1
                    print(id + " it is copied!")
                    shutil.copyfile(original, target)
                else: print(id + " NOT copied!")
    return match

def rename_jpg_images(path): #ID-YYYYMMDDThhmmss_iriis.jpg
    files=os.listdir(path)
    for file in files:
        old_name = path + "\\" + file
        splitted=file.split("_")
        if(len(splitted)==8):
            id=splitted[0]
            year=splitted[1]
            month=splitted[2]
            day=splitted[3]
            hour=splitted[4]
            min=splitted[5]
            sec=splitted[6]
            new_name=path+ "\\" +id+"-"+year+month+day+"T"+hour+min+sec+"_iriis.jpg"
            print(old_name,new_name)
            os.rename(old_name, new_name)


#matches=matching_mapping(krdi_ids,path,triplets_path)
#print(matches)

#rename_jpg_images(triplets_path)
def get_ids2(path):
    files = os.listdir(path)
    ids=[]
    for file in files:
        id= file.split("-")[0]
        #print(id)
        ids.append(id)
    return ids

ids=get_ids2(triplets_path)


def copy_triplets_from_krdis(krdi_path,ids,triplets_path):
    krdis = os.listdir(krdi_path)
    for krdi in krdis:
        id_krdi=krdi.split("_")[0]
        if id_krdi in ids:
            original = krdi_path + "\\" + krdi
            target = triplets_path + "\\" + krdi
            shutil.copyfile(original,target)

#copy_triplets_from_krdis(krdi_path,ids,triplets_path)




