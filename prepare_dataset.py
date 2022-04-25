import os
import shutil
from os import listdir
from Configs import SharedConfigurations

############################# FUNCTIONS : : : ############################################################
##########################################################################################################
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



def get_iriis_ids(path): # ids from ***IRIIS***
    files = os.listdir(path)
    ids=[]
    for file in files:
        id= file.split("_")[0].split("-")[0]
        if(id!="Thumbs.db"):
            #print(id)
            ids.append(id)
    return ids



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



############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

conf=SharedConfigurations()


# 4. STEP rename converted ".krdi" files (now we have .krdi + 2 png-s (depth and color image) :
converted_krdis=conf.converted_krdis
rename_converted_krdi_files(converted_krdis)


# 5. STEP - create FINAL DATASET  :

# 5.1. : move all iriis files in FINAL DATASET FOLDER
iriis_dest=conf.iriis_dest
iriis_root=conf.iriis_folder

#number_of_iriis_images=move_iriis_files_to_dest_folder(iriis_root,iriis_dest)

# 5.2. : rename all iriis files (naming convention is :  #ID-YYYYMMDDThhmmss_iriis.jpg)

#renamed=rename_iriis_files(iriis_dest)
#print(renamed)  ############################################# 21318 files iriis files !

# 5.3. copy all matching files from sirius to FINAL DATASET (*matching* => ID matces ID from iriis)
sirius_root=conf.sirius_dest
sirius_dest=conf.sirius_dest

#iriis_ids=get_iriis_ids(iriis_dest)
#matches=find_and_extract_matches(iriis_ids,sirius_root,sirius_dest)
#print(matches)

# 5.4 rename all sirius files (naming convention: ID-YYYYMMDDThhmmss_sirius.krdi + ID-YYYYMMDDThhmmss_sirius-depth.png + ID-YYYYMMDDThhmmss_sirius-color.png)
#rename_converted_krdi_files(sirius_dest)
#times=get_times(iriis_dest)
#add_time2name(times,sirius_dest)


# Continue with JSON_edditing.py script

########################## FIX ::: iriis umschlichtung images have 11 characters long ids but sirius images have 10 chars long ids (umschlichtung case)
### help yourself with this code :
'''
import os
import shutil
iriis_path=r"G:\Matea\FINAL_DATASET\wuerth_iriis" #longer
sirius_path=r"G:\Matea\krdis4_renamed" #shorter
sirius_new_path=r"G:\Matea\FINAL_DATASET\wuerth_sirius"
matching_ums_path=r"G:\Matea\umschlihtung_filtered_images"
def get_ums_iriis(path):
    ims=os.listdir(path)
    ids=[]
    cnt=0
    for im in ims:
        id=im.split("_")[0].split("-")[0]
        if(id[0]=="4"):
            #print(id, " ; ", len(id)) # len=11
            id=id[:-1]     ##################### IMPORTANT
            #print(id, " ; ", len(id))
            ids.append(id)
            cnt+=1
    print(cnt)
    return ids

def get_ums_sirius(path):
    ims=os.listdir(path)
    ids=[]
    cnt=0
    for im in ims:
       exstension = im[-9:]
       if(exstension=="color.png" and im[0]=="4"):
         id=im.split("_")[0]
         ids.append(id)
         #print(id , " ; " , len(id)) 'len=10
         cnt+=1
    print(cnt)
    return ids


def get_matches(ids_iriis,ids_sirius):
    matches=[]
    cnt=0
    for id in ids_sirius:
        if id in ids_iriis:
            #print(id)
            matches.append(id)
            cnt+=1
    print(cnt)
    return matches


def extract_matching_files(path_krdis,new_path,matches):
    krdis=os.listdir(path_krdis)
    cnt=0
    for krdi in krdis:
        id=krdi.split("_")[0]
        if id in matches:
            old_path=path_krdis + "\\" + krdi
            dest_path=new_path + "\\" + krdi
            #print(old_path, "****",dest_path)
            shutil.copy(old_path,dest_path)
            cnt+=1
            #print(cnt)


def get_times(path):
    files = os.listdir(path)
    times = {}
    for file in files:
        extension = file[-4:]
        if (extension == ".jpg"):
            id = file.split("-")[0].split("_")[0]
            id = id[:-1]
            time =file.split("_")[0].split("-")[1]
            times[id]=time
    return times

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

def copy_ums(ums_path,sirius_new_path):
    files=os.listdir(ums_path)
    cnt=0
    for file in files:
        old_path=ums_path + "\\" + file
        dest_path=sirius_new_path + "\\" + file
        cnt+=1
        shutil.copy(old_path,dest_path)
    print(cnt)


#ums_iriis=get_ums_iriis(iriis_path) # 3173
#ums_sirius=get_ums_sirius(sirius_path) # 2007

#matches=get_matches(ums_iriis,ums_sirius) #1730

#extract_matching_files(sirius_path,matching_ums_path,matches)
#times=get_times(iriis_path)
#add_time2name(times,matching_ums_path)

# now just copy all of these files to FINAL_DATASET/wueth_sirius
#######copy_ums(matching_ums_path,sirius_new_path)


'''
