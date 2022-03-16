import os
import shutil
from os import listdir
from os.path import isfile, join

krdi_path = r"G:\Matea\krdis1"
krdi_path2=r"G:\Matea\krdis2"
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



ids=add_ids(krdi_path)
filtered_ids=checkIfDuplicates(ids)
copy_filtered_krdi(krdi_path,krdi_path2,filtered_ids)
