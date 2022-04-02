# task
##FINAL DATASET :
##1) iriis -- exactly the same number of images as I can find in .json file
##2) sirius -- just matches !

import os
import shutil
from os import listdir
from os.path import isfile, join
import json
from Configs_parsing import SharedConfigurations

conf=SharedConfigurations()

iriis_dest=conf.iriis_dest
iriis_root=conf.iriis_folder

def move_iriis_files_to_dest_folder(path,dest):
    boxes=os.listdir(path)
    cnt=0
    for dir in boxes:
        file_path = path + "\\" + dir
        files=os.listdir(file_path)
        for file in files:
                id=file.split("_")[0]
                original = file_path + "\\" + file
                target = dest + "\\" + str(file)
                print(original + "==> " + target)
                shutil.copyfile(original, target)
                cnt+=1

#number_of_iriis_images=move_iriis_files_to_dest_folder(iriis_root,iriis_dest)


def rename_iriis_files(path): #ID-YYYYMMDDThhmmss_iriis.jpg
    files=os.listdir(path)
    cnt=0
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
            #print(old_name,new_name)
            print(cnt)
            cnt+=1
            os.rename(old_name, new_name)

    return cnt

#renamed=rename_iriis_files(iriis_dest)
#print(renamed)  ############################################# 21318 files iriis files !

# check if it is same number of files as in .json file :
def count_ids_in_jsonFile(input_file):
    json_decode = json.load(input_file)
    classes={}
    cnt=0
    for filename in json_decode["_via_img_metadata"]:
        id = json_decode["_via_img_metadata"][filename]["filename"].split("/")[1].split("_")[0]
        print(id,cnt)
        cnt+=1
    return cnt


#input_file=open('wuerth_iriis.json', 'r')
#num_of_ids_in_jsonFile=count_ids_in_jsonFile(input_file)    #########20716 files in .json


sirius_root=conf.sirius_folder_krdis_converted
sirius_dest=conf.sirius_dest

def get_iriis_ids(path): # ids from ***IRIIS***
    files = os.listdir(path)
    ids=[]
    for file in files:
        id= file.split("_")[0].split("-")[0]
        if(id!="Thumbs.db"):
            #print(id)
            ids.append(id)
    return ids

iriis_ids=get_iriis_ids(iriis_dest)

def find_and_extract_matches(ids,path,sirius_dest):
    files=os.listdir(path)
    match=0
    for file in files:
            id=file.split("_")[1].split(".")[0]
            #print(id)
            original = path + "\\" + file
            target = sirius_dest + "\\" + str(file)
            if id in ids:
                match+=1
                #shutil.copyfile(original, target)
            else: print(id + " NOT copied!")
    return match

#matches=find_and_extract_matches(iriis_ids,sirius_root,sirius_dest)
#print(matches)



#rename converted "krdi" files (now we have .krdi + 2 png-s (depth and color image) :
def rename_converted_krdi_files(path):
    files = os.listdir(path)
    cnt=0
    for file in files:
        old_name=path+"\\"+file
        id= file.split(".")[0]
        depth_or_image_stamp=id[-4:]
        id= id.split("_")[1]
        if (depth_or_image_stamp == "_000"):
            new_name = path + "\\" + id + "_sirius-color.png"
            #print(old_name, new_name)
            os.rename(old_name, new_name)
        elif (depth_or_image_stamp == "_001"):
            new_name = path + "\\" + id  + "_sirius-depth.png"
            #print(old_name, new_name)
            os.rename(old_name, new_name)
        elif (depth_or_image_stamp != "_000" and depth_or_image_stamp != "_001" ):
            new_name = path + "\\" + id + "_sirius.krdi"
            #print(old_name, new_name)
            os.rename(old_name, new_name)
        cnt+=1


#rename_converted_krdi_files(sirius_dest)



def get_times(path):
    krdis = os.listdir(path)
    times = {}
    for krdi in krdis:
        extension = krdi[-4:]
        if (extension == ".jpg"):
            id = krdi.split("-")[0].split("_")[0]
            time = krdi.split("_")[0].split("-")[1]
            times[id]=time
    return times

times=get_times(iriis_dest)



def add_time2name(times,path):
    files = os.listdir(path)
    cnt=0
    for file in files:
        extension = file[-4:]
        if (extension != ".jpg"):
            old_name=path+"\\"+ file
            id = file.split("-")[0].split("_")[0]
            if id in times:
               #print(id, times[id])
               exten=file.split("_")[1]
               name=id+"-"+times[id]+"_"+exten
               new_name=path+"\\"+name
               #print(old_name + " ===> " +new_name)
               os.rename(old_name,new_name)
               cnt+=1
               if(cnt%1000==0): print(cnt)


add_time2name(times,sirius_dest)
