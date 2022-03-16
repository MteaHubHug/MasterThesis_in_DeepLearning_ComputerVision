import os
import shutil
from os import listdir
from os.path import isfile, join


#path = r"E:\Matea\wuerth_boeheimkirchen"
krdi_path = r"G:\Matea\krdis3"
krdis = os.listdir(krdi_path)
#rename krdi files :
for krdi in krdis:
    old_name=krdi_path+"\\"+krdi
    id_krdi = krdi.split(".")[0]
    depth_or_image_stamp=id_krdi[-4:]
    id_krdi= id_krdi.split("_")[1]
    if (depth_or_image_stamp == "_000"):
        new_name = krdi_path + "\\" + id_krdi + "_sirius-color.png"
        print(new_name)
        os.rename(old_name, new_name)
    elif (depth_or_image_stamp == "_001"):
        new_name = krdi_path + "\\" + id_krdi + "_sirius-depth.png"
        os.rename(old_name, new_name)
    elif (depth_or_image_stamp != "_000" and depth_or_image_stamp != "_001" ):
        new_name = krdi_path + "\\" + id_krdi + "_sirius.krdi"
        os.rename(old_name, new_name)




####################   and depth_or_image_stamp != "olor" and depth_or_image_stamp != "rius" and depth_or_image_stamp != "epth"