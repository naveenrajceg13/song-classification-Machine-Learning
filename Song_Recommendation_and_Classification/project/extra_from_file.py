import csv
with open("sample.csv",'r') as my_file:
                try:
                    s=my_file.read()
                except:
                    pass
                

#b = open('sample.csv', 'w', newline='')
#a = csv.writer(b)
def extract_fur():
    str=s.split("\n")
    total_length=len(str)
    count=1
    data=[[]]
    class_name=[]
    file_name=[]
    #print(total_length)
    for each in str:
        values=each.split(",")
        #print(count,total_length)
        if(count<total_length-2):
            if(count!=1):
                temp=[]
                data.append(temp)
            for each1 in values:
                data[count-1].append(float(each1))
        if(count==total_length-2):
            #print("last before",count)
            for each1 in values:
                class_name.append(int(each1))
        if(count>=total_length-1):
            #print("last",count)
            for each1 in values:
                file_name.append(each1)
        count=count+1
            
    #print("data",data)
    #print("class",class_name)
    #print("file",file_name)
    
    return data,class_name,file_name
    #a.writerows(file_name)
    #b.close()