Song Classification and recommendation system

Platform and Software needed to be installed: 
1. All platform supported 
2. Python 3.5 need to be installed.
Python Packages need to be installed:
i. Sklearn
ii. Numpy
iii. Csv
iv. Urllib
v. Urllib3
vi. pyAudioAnalysis
vii. matplotlib
viii. pydub
ix. pyqt4
Running:
1) First we need to convert MP3 files to .wav format. For this we need to run converttowave.py

File path as the changes as the needed one. Each path is the folder that contains mp3 songs. 
2) Convert .wav to frequency array. This will be obtained by running extractfeaturefromfile.py

Similarly we need to give, converted .wav files format. Each Class will be given as one folder so that it will be easy to name the class. 

3)  From the .wav files we need to generate csv file, so that it will not take long time while running for 2nd time. Only for the first time we need to run this from the 2nd time we can directly run another program that extract feature from this excel files. We need to run the program runningalgo.py
 
Similarly file path has to be changes, as per the requirement.



4) Main_Class.py is the class that we need to run now, this will first extract features from the file. 
4.1) extra_from_file.py. This will extract features from file and pass as 2D matrix. 

File path of excel has to be taken care. 
4.2) BuildModels.py. This will build the models that we can use in your GUI, once the model is fitted it will take only less time to classify or cluster. 


 Steps to run Main_Class.py:
1) Open Commend prompt in the path where project is there. 
2) Get the python installed path. 
(eg: C:\Users\NAVE\AppData\Local\Programs\Python\Python35-32\python.exe) 
3) Run filepath/python.exe<<space>> Main_Class.py




4) GUI will be opened. 


Selecting a song will suggest us similar song as playing. 
Select class from computer or from the list present and click get class to classify the song. 
Similarly get probability will give probability of it belongs to all the clusters. 
Selecting the particular cluster will list songs in that cluster. 






