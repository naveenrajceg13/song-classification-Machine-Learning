'''
Created on Apr 22, 2016

@author: NAVE
'''
import os;
import xlwt;
import csv;
import math
def Getfilesinfolder(path):
    list=[]
    for dir_entry in os.listdir(path):
        dir_entry_path = os.path.join(path, dir_entry)
        #print(dir_entry_path)
        if os.path.isfile(dir_entry_path):
            list.append(dir_entry_path)
    return list;
def getfilesintofile(list_1,list_2,list_3,count,class_val,sheet,dist):
    list_count=0
    xnan=float('nan')  
    for files in list_1:
            col_count=0
            sub_count=0
            bool=True
            
            with open(list_1[list_count],'r') as my_file:
                try:
                    s=my_file.read()
                except:
                    pass
                str1=s.split("\n")
                for each in str1:
                    each=each.replace('s',' ')
                    str2=each.split(" ")
                    scount=0
                    for each2 in str2:
                        scount=scount+1
                    if(count==1):
                        sheet.write(count-1,col_count,str2[scount-2]+"_"+str(col_count))
                        #dist[count-1,col_count]=str2[scount-2]+"_"+str(col_count)
                        dist[0].append(str2[scount-2]+"_"+str(col_count))
                    sheet.write(count,col_count,float(str2[scount-1]))
                    value1=float(str2[scount-1])
                    if(math.isnan(float(str2[scount-1]))):
                            value1=-1
                    dist[count].append(value1)
                    
                    col_count=col_count+1  
                    #print(str2[scount-2],str2[scount-1])
                    if(scount==4):
                        if(count==1):
                            dist[0].append(str2[0]+"_"+str(col_count))
                            #sheet.write(count-1,col_count,str2[0]+"_"+str(col_count))
                        sheet.write(count,col_count,float(str2[scount-1]))
                        #dist[count,col_count]=float(str2[scount-1])
                        dist[count].append(float(str2[1]))
                        col_count=col_count+1
                        #print(str2[0],str2[1])
            
            with open(list_2[list_count],'r') as my_file:
                try:
                    s=my_file.read() 
                except:
                    pass
                str1=s.split("\n")
                ncount=1
                for each in str1:
                    try:
                        value2=float(each)
                        if(math.isnan(float(each))):
                            value2=-1
                        dist[count].append(value2)
                        if(count==1):
                            dist[0].append("MFCC"+str(ncount))
                        ncount=ncount+1
                        if(ncount>=15):
                            break
                    except:
                        pass
            
            with open(list_3[list_count],'r') as my_file:
                try:
                    s=my_file.read()
                except:
                    pass
                str1=s.split("\n")
                ccount=1
                num_cols=0
                header=str1[0]
                header=header.replace(" ",",")
                header=header.split(',')
                #print(header)
                header_count=0
                total_header=len(header)
                chroma_header=1
                for each in str1:
                    each=each.split(' ')
                    for each1 in each:
                        #print(each1)
                        #print(len(each1))
                        if(len(each1)==0):
                            continue
                        if(ccount>1):
                            if(count==1):
                                #print((header[header_count]))
                                #sheet.write(count-1,col_count,"asdas")
                                #dist[count-1,col_count]=header[header_count]
                                dist[0].append(str(header[header_count])+"_"+str(int(chroma_header)))
                                header_count=header_count+1
                                if(header_count>=total_header-1):
                                    header_count=0
                                chroma_header=chroma_header+1
                            #print(col_count)    
                            #sheet.write(count,col_count,1231)
                            #dist[count,col_count]=each1
                            value3=float(each1)
                            if(math.isnan(float(each1))):
                                value3=-1
                            dist[count].append(value3)
                            col_count=col_count+1
                            num_cols=num_cols+1
                    ccount=ccount+1
                                
                        #print(each1)
                    #print("\n")
            
            dist[count].append(class_val)
            if(count==1):
                dist[0].append(("Class"))
            count=count+1
            temp1=[]
            dist.append(temp1)
            #print(len(dist[count-1]))
            list_count=list_count+1
            
            
    return count
def function_feature(): 
    dist=[[]]
    temp=[]
    dist.append(temp)
    count=1
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet 1")
    list_1=Getfilesinfolder("C:\project\Features\A\Rock")
    list_2=Getfilesinfolder("C:\project\Features\B\Rock")
    list_3=Getfilesinfolder("C:\project\Features\C\Rock")
    count=getfilesintofile(list_1,list_2,list_3,count,1,sheet1,dist)
    list_1=Getfilesinfolder("C:\project\Features\A\Jazz")
    list_2=Getfilesinfolder("C:\project\Features\B\Jazz")
    list_3=Getfilesinfolder("C:\project\Features\C\Jazz")
    count=getfilesintofile(list_1,list_2,list_3,count,2,sheet1,dist)
    list_1=Getfilesinfolder("C:\project\Features\A\Hip-Hop")
    list_2=Getfilesinfolder("C:\project\Features\B\Hip-Hop")
    list_3=Getfilesinfolder("C:\project\Features\C\Hip-Hop")
    count=getfilesintofile(list_1,list_2,list_3,count,3,sheet1,dist)
    list_1=Getfilesinfolder("C:\project\Features\A\EDM")
    list_2=Getfilesinfolder("C:\project\Features\B\EDM")
    list_3=Getfilesinfolder("C:\project\Features\C\EDM")
    count=getfilesintofile(list_1,list_2,list_3,count,4,sheet1,dist)
    list_1=Getfilesinfolder("C:\project\Features\A\country")
    list_2=Getfilesinfolder("C:\project\Features\B\country")
    list_3=Getfilesinfolder("C:\project\Features\C\country")
    count=getfilesintofile(list_1,list_2,list_3,count,5,sheet1,dist)
    book.save("trial.xls")
    b = open('data_change_frequency.csv', 'w', newline='')
    a = csv.writer(b)
    a.writerows(dist)
    b.close()
    return dist