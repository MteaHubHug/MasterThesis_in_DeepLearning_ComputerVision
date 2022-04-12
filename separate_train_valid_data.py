
#### useful code for separating into training and validation datasets


import os
import shutil

from Configs import SharedConfigurations
configs=SharedConfigurations()

def get_train_files(lines):
    train_files = []
    for ctr, filnam in enumerate(lines):
        filename = filnam.split("/")[6][0:-1]
        #print(filename)
        train_files.append(filename)
    return  train_files


def get_valid_files(lines):
    valid_files = []
    for ctr, filnam in enumerate(lines):
        filename = filnam.split("/")[6][0:-1]
        print(filename)
        valid_files.append(filename)
    return  valid_files



def copy_train_files(train_files,ok_folder,nok_folder,train_folder):
    cnt=0
    ok_images=os.listdir(ok_folder)
    for image in ok_images:
        if image in train_files:
            original=ok_folder+"\\" + image
            dest=train_folder+"\\OK" + "\\" + image
            shutil.copy(original,dest)
            print(image," OK + train ", cnt)
            cnt+=1

    nok_images=os.listdir(nok_folder)
    for image in nok_images:
        if image in train_files:
            original = nok_folder + "\\" + image
            dest=train_folder+"\\NOK" + "\\" + image
            shutil.copy(original, dest)
            print(image, " NOK + train", cnt)
            cnt+=1

    return cnt



def copy_valid_files(valid_files,ok_folder,nok_folder,valid_folder):
    cnt=0
    ok_images=os.listdir(ok_folder)
    for image in ok_images:
        if image in valid_files:
            original=ok_folder+"\\" + image
            dest=valid_folder+"\\OK" + "\\" + image

            shutil.copy(original,dest)
            print(image," OK - valid ", cnt)
            cnt+=1

    nok_images=os.listdir(nok_folder)
    for image in nok_images:
        if image in valid_files:
            original = nok_folder + "\\" + image
            dest=valid_folder+"\\NOK" + "\\" + image

            shutil.copy(original, dest)
            print(image, " NOK - valid ", cnt)
            cnt+=1

    return cnt

train_file_names = configs.train_images_name_file
valid_file_names = configs.validation_images_name_file

ok_folder = configs.ok_folder
nok_folder = configs.nok_folder
train_folder = configs.orig_train_folder
valid_folder= configs.orig_valid_folder

#with open(train_file_names) as f:
#    L = f.readlines()

#trains=get_train_files(L)
#copied_train=copy_train_files(trains,ok_folder,nok_folder,train_folder)
#print(copied_train)


with open(valid_file_names) as v:
    V = v.readlines()

valids=get_valid_files(V)
copied_valid=copy_valid_files(valids,ok_folder,nok_folder,valid_folder)
print(copied_valid)



