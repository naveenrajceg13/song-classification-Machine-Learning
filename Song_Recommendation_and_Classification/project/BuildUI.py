import sys
from PyQt4 import QtGui,QtCore
from msilib.schema import CheckBox
from Cython.Plex.Lexicons import State
import pyglet
import os
#import time, wave, pymedia.audio.sound as sound

class Window(QtGui.QMainWindow):
    
    def __init__(self):
        
        pass#self.home()
    
    def home(self):
        super(Window,self).__init__()
        self.setGeometry(50,50,1000,600)
        self.setWindowTitle("Song Classification")
        self.setWindowIcon(QtGui.QIcon('D:\python.png'))
        extractAction=QtGui.QAction("&Exit the Window",self)
        extractAction.setShortcut("Ctrl+Q")
        extractAction.setStatusTip('Leave the app')
        extractAction.triggered.connect(self.close_application)
        self.statusBar()
        mainMenu=self.menuBar()
        fileMenu=mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction)
        self.styleChoice=QtGui.QLabel("Select a song",self)
        self.comboBox=QtGui.QComboBox(self)
        self.comboBox.move(450,200)
        self.comboBox.activated[str].connect(self.style_choice)
        self.styleChoice.move(350,200)
        self.show()
    
    def color_picker(self):
        color=QtGui.QColorDialog.getColor()
        self.styleChoice.setStyleSheet("QWidget { background-color: %s}" %color.name())
        
        
    
    def style_choice(self,text):
        self.styleChoice.setText(text)
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create(text))
        
    def download(self):
        self.completed=0
        
        while self.completed<100:
            self.completed += 0.0001
            self.progress.setValue(self.completed)
        
    def close_application(self):
        choice=QtGui.QMessageBox.question(self,'Extract!',
                                          "get into the exit?",QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
        if choice==QtGui.QMessageBox.Yes:
            print("extrating naaaaaaaaaaaaa")
            sys.exit()
        else:
            pass
        
    def enlarge_window(self,state):
        if state == QtCore.Qt.Checked:
            self.setGeometry(50,50,1000,600)
        else:
            self.setGeometry(50,50,500,300)
    
    def listallsongs(self,song_list):
        
            self.comboBox.addItem("sasasasasdasdasd")    
        

app=QtGui.QApplication(sys.argv)
GUI=Window()       
def run_window():
    sys.exit(app.exec_())
def function_UI():            
        run_window()
        return GUI
        #listallsongs(song_list)
        
def listallfiles(song_list):
    GUI.listallsongs(song_list)

