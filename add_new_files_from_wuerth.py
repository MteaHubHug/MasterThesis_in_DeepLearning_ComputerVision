import os
import shutil
from os import listdir
from Configs import SharedConfigurations


conf=SharedConfigurations()

path_new_files_folders=r"G:\Matea\VK9_AT-W"
path_new_files_ok=r"G:\Matea\VK9_AT-W\VK9_OK"
path_new_files_nok=r"G:\Matea\VK9_AT-W\VK9_NOK"
path_new_files_damaged=r"G:\Matea\VK9_AT-W\VK9_DAMAGED"
dest_matches=r"G:\Matea\umschlichtung_filtered_images_iriis"
path_sirius_images=r"G:\Matea\umschlihtung_filtered_images"

def rename_iriis_files(path): #ID-YYYYMMDDThhmmss_iriis.jpg
    files=os.listdir(path)
    cnt=0
    for file in files:
        f=file.split("_")
        old_filename=path+"\\"+file
        if(len(f)==7):
            classy = f[6].split(".")[0]
            y=f[0]
            m=f[1]
            d=f[2]
            h=f[3]
            min=f[4]
            s=f[5]
            stamp=y+m+d+"T"+h+min+s
            new_filename=path + "\\" + stamp+"_iriis.jpg"
            cnt+=1
            print(cnt)
            os.rename(old_filename,new_filename)



#rename_iriis_files(path_new_files_nok)

def make_id_time_dict_sirius(path_sirius):
    sirius_ims=os.listdir(path_sirius)
    id_timestamp_sirius={}
    for sir in sirius_ims:
        ex=sir[-9:]
        if(ex=="color.png"):
            id_timestamp=sir.split("_")[0]
            id=id_timestamp.split("-")[0]
            timestamp=id_timestamp.split("-")[1]
            id_timestamp_sirius[timestamp]=id # id
    return id_timestamp_sirius


def find_times(path_iriis_ok,path_iriis_nok):
    iriis_oks=os.listdir(path_iriis_ok)
    iriis_noks=os.listdir(path_iriis_nok)
    oks=[]
    noks=[]
    for ok in iriis_oks:
        time=ok.split("_")[0]
        oks.append(time)
    for nok in iriis_noks:
        time=nok.split("_")[0]
        noks.append(time)
    return  oks, noks


def find_matches(oks,noks,dict):
    for time in dict:
        if time in oks:
            print(noks)

sirius_dict=make_id_time_dict_sirius(path_sirius_images)
oks, noks= find_times(path_new_files_ok,path_new_files_nok)
find_matches(oks,noks,sirius_dict)
