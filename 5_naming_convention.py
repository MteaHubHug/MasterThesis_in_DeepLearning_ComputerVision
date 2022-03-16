import os
import shutil
krdi_path = r"G:\Matea\triplets"

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

times=get_times(krdi_path)



def add_time2name(times,path):
    krdis = os.listdir(path)
    for krdi in krdis:
        extension = krdi[-4:]
        if (extension != ".jpg"):
            old_name=path+"\\"+ krdi
            id = krdi.split("-")[0].split("_")[0]
            if id in times:
               #print(id, times[id])
               exten=krdi.split("_")[1]
               name=id+"-"+times[id]+"_"+exten
               new_name=path+"\\"+name
               #print(old_name + " ***** " +new_name)
               os.rename(old_name,new_name)



add_time2name(times,krdi_path)
