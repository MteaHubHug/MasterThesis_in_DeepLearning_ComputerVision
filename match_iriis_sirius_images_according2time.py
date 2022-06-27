import os
import shutil
import csv
import math
import datetime
disk_path=r"E:\backup"
krdi_folder=r"E:\krdi_files"

iriis_folder=r"E:\IRIISxSIRIUS\IRIIS_matches"
sirius_folder=r"E:\IRIISxSIRIUS\SIRIUS"
matches_folder= r"E:\IRIISxSIRIUS\IRIISandSIRIUS_matches"
def read_csv_files(file_path):
    f = open(file_path)
    flag=0
    csvreader = csv.reader(f)

    infos=[]
    for row in csvreader:
        if (row[0][0] == "0"):
            r = row[0]
            excel_id = r[0:10]
            timestamp = r[37:56]
            id = r[57:67]

            if (timestamp[0:4] == "2022"):
                year=timestamp[0:4]
                month=timestamp[5:7]
                day=timestamp[8:10]
                hours=timestamp[11:13]
                mins=timestamp[14:16]
                secs=timestamp[17:19]
                new_timestamp=year+month+day + "T" + hours + mins + secs
                flag=1
                #print(excel_id, " *** ", timestamp, " *** ", id)
                new_name=excel_id + "_" + id + "_" + new_timestamp + ".krdi"
                infos = [excel_id, timestamp, id, new_name, new_timestamp]


    f.close()
    return flag, infos

def extract_timestamp_from_csv_and_concat_w_krdi(path):
    dates=os.listdir(path)
    cnt=0
    excel_ids=[]
    for date in dates:
        if(date[0]=="2"):
            date_folder=path + "\\" + date
            ids=os.listdir(date_folder)
            for id in ids:
                id_folder=date_folder + "\\" + id
                files=os.listdir(id_folder)
                for file in files:
                    if(file[-4:]==".csv"):
                        krdi_file=file[:-4] + ".krdi"
                        krdi_path= id_folder + "\\" + krdi_file
                        excel_id= file[:-4]
                        excel_ids.append(excel_id)
                        file_path = id_folder + "\\" + file
                        flag , infos =read_csv_files(file_path)
                        if(flag==1):
                            cnt+=1
                            new_name=infos[3]
                            new_path= id_folder + "\\" + new_name
                            print(krdi_path, " *** ", new_path)
                            #os.rename(krdi_path,new_path)


    print(cnt)
    return excel_ids

#excel_ids=extract_timestamp_from_csv_and_concat_w_krdi(disk_path)     # 5235 normal / 6106 . csv file (excel)  / 9395 sirius images (.krdi)

def extract_krdi_files(path,dest):
    dates=os.listdir(path)
    cnt=0
    for date in dates:
        if(date[0]=="2"):
            date_folder=path + "\\" + date
            ids=os.listdir(date_folder)
            for id in ids:
                id_folder=date_folder + "\\" + id
                files=os.listdir(id_folder)
                for file in files:
                    if(file[-4:]=="krdi"):
                       if(len(file)>27):
                           #print(file)
                           old_file_path=id_folder + "\\" + file
                           new_file_path= dest + "\\" + file
                           #print(old_file_path, " *** ", new_file_path)
                           shutil.move(old_file_path,new_file_path)
                           if(cnt%100==0): print(cnt)
                           cnt+=1


    print(cnt)


#extract_krdi_files(disk_path,dest_path)


def rename_iriis(path):
    files=os.listdir(path)
    for file in files:
        if(file[0]=="2"):

          splitted=file.split("_")
          year=splitted[0]
          month=splitted[1]
          day=splitted[2]
          hour=splitted[3]
          min=splitted[4]
          sec=splitted[5]
          timestamp= year + month + day + "T" + hour + min + sec
          old_name= path + "\\" + file
          new_name= path + "\\" + timestamp + ".jpg"
          #print(old_name, " *** ", new_name)
          #os.rename(old_name,new_name)


#rename_iriis(iriis_folder)

def sirius_times(path):
    files=os.listdir(path)
    infos=[]
    for file in files:
        splitted=file.split("_")
        excel_id=splitted[0]
        id=splitted[1]
        timestamp=splitted[2].split(".")[0]
        #print(excel_id, " *** ", id, " *** ", timestamp)
        info= [id,timestamp,excel_id]
        infos.append(info)
    return infos

def iriis_times(path):
    files=os.listdir(path)
    infos=[]
    for file in files:
        if(file[0]=="2"):
          timestamp=file.split(".")[0]
          timestamp=timestamp[:-9]
          #print(timestamp)
          infos.append(timestamp)
    return infos

#sirius_infos=sirius_times(krdi_folder)
#iriis_infos=iriis_times(iriis_folder)
#####################################################
###########################################################
krdis_final_folder=r"E:\MATEA_IVIIxWUERTH\krdis_final"

def find_mod_time(path):
    files=os.listdir(path)
    cnt=0
    mod_times={}
    for file in files:
        if(file!="Thumbs.db"):
            if(file[-4:]=="krdi"):
                file_path=path+"\\"+file
                timemod=os.path.getmtime(file_path)
                dt_c = datetime.datetime.fromtimestamp(timemod)
                dt_c=str(dt_c)
                dtc=dt_c[0:19]
                y=dtc[0:4]
                m=dtc[5:7]
                d=dtc[8:10]
                h=dtc[11:13]
                min=dtc[14:16]
                s=dtc[17:19]
                timestamp= y + m + d + "T" + h + min + s
                id=file.split("_")[0] + "_" + file.split("_")[1]
                mod_times[id]=timestamp
    return mod_times

#####mod_times=find_mod_time(krdis_final_folder)

def rename_sirius_files(path,mod_times):
    files=os.listdir(path)
    cnt=0
    for file in files:
        id=file.split("_")[0] + "_" + file.split("_")[1]
        if id in mod_times:
            old_name=path + "\\" + file
            sufix=file.split("_")[2]
            new= file.split("_")[0] + "_" + mod_times[id] + "_" + sufix
            new_name= path + "\\" + new
            os.rename(old_name, new_name)
            cnt+=1
            if(cnt%100==0): print(cnt)
    print(cnt)



######rename_sirius_files(krdis_final_folder, mod_times)
######################################################################
########################################################################
'''def find_matches(sirius,iriis):
    for sir in sirius:
        sirius_time=sir[1][:-9]
        print("sirius : ", sirius_time)
        #if(sirius_time in iriis):
        #    print(sirius_time)
    for iri in iriis:
        print("iriiis : ", iri)

###find_matches(sirius_infos,iriis_infos)'''


def get_matches(path_iriis,path_sirius):
    irisi=os.listdir(path_iriis)
    siriusi=os.listdir(path_sirius)
    cnt=0
    iriis_times=[]
    sirius_times=[]
    for iris in irisi:
        timestamp_iriis=iris.split(".")[0]
        iriis_times.append(timestamp_iriis)

    for sirius in siriusi:
        timestamp_sirius=sirius.split("_")[1]
        sirius_times.append(timestamp_sirius)
    matches={}
    for iris in iriis_times:
        y_i=int(iris[0:4])
        m_i=int(iris[4:6])
        d_i=int(iris[6:8])
        h_i=int(iris[9:11])
        min_i=int(iris[11:13])
        s_i=int(iris[13:15])
        dt_iriis=datetime.datetime(y_i,m_i,d_i,h_i,min_i,s_i)
        for sirius in sirius_times:
            y_s = int(sirius[0:4])
            m_s = int(sirius[4:6])
            d_s = int(sirius[6:8])
            h_s = int(sirius[9:11])
            min_s = int(sirius[11:13])
            s_s = int(sirius[13:15])
            dt_sirius= datetime.datetime(y_s,m_s,d_s,h_s,min_s,s_s)
            tdelta = dt_iriis - dt_sirius
            tdelta=str(tdelta)
            if(tdelta=="0:00:00" or tdelta=="0:00:01" or tdelta=="0:00:02" or tdelta=="0:00:03" or tdelta=="0:00:04" or tdelta=="0:00:05"  ):
              cnt+=1
              date_iriis=iris[0:8]
              date_sirius=sirius[0:8]
              if(date_sirius==date_iriis):
                  matches[iris]=sirius


    print(cnt)
    return matches


#matches=get_matches(iriis_folder,sirius_folder)
def rename_iriis_images(matches,path_iriis,path_sirius,dest):
    cnt=0
    iriises = os.listdir(path_iriis)
    siriuses=os.listdir(path_sirius)
    for iris in iriises:
        timestamp_iriis=iris[:-4]
        if timestamp_iriis in matches:
            sirius_timestamp=matches[timestamp_iriis]
            for sirius in siriuses:
                timestamp_id=sirius.split("_")[1]
                if(timestamp_id==sirius_timestamp):
                    id=sirius.split("_")[0]
                    old_name_iriis = iriis_folder + "\\" + iris
                    new_name_iriis= id + "_" + sirius_timestamp +"_iriis.jpg"
                    new_name_iriis = iriis_folder + "\\" + new_name_iriis
                    print(old_name_iriis, " *** " , new_name_iriis)
                    shutil.move(old_name_iriis,new_name_iriis)
    print(cnt)




#rename_iriis_images(matches,iriis_folder,sirius_folder,matches_folder)


def copy_matches(path_iriis,path_sirius,dest):
    iriises=os.listdir(path_iriis)
    siriuses=os.listdir(path_sirius)
    cnt=0
    for iriis in iriises:
        match_iriis=iriis[0:26]
        for sirius in siriuses:
            match_sirius=sirius[0:26]
            if (match_iriis==match_sirius):
               cnt+=1
               old_iriis= path_iriis + "\\" + iriis
               new_iriis= dest + "\\" + iriis
               old_sirius= path_sirius + "\\" + sirius
               new_sirius= dest + "\\" + sirius
               shutil.copy(old_iriis,new_iriis)
               shutil.copy(old_sirius,new_sirius)

    print(cnt)
copy_matches(iriis_folder,sirius_folder,matches_folder)


