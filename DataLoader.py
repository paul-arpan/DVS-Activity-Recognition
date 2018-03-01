from __future__ import print_function, division
import cv2
import os
import numpy as np
import av
import re


class videoDataset():
    """Dataset Class for Loading Video"""

    def __init__(self, path):
        
        self.rootDir = path
        name=[]
        file=[]
        for _, dirnames, filenames in os.walk(path):
            name.append(dirnames)
            file.append(filenames)
        name = name[0]

        newpath = []

        for i in range(len(name)):    
            for files in file[i+1]:
                pathn = path + name[i]+'/' + files
                newpath.append(pathn)
                
        self.sequenceLength = 5
        self.classList=['Basketball','Biking','Diving','GolfSwing','HorseRiding','SoccerJuggling','Swing','TennisSwing','TrampolineJumping','VolleyballSpiking','WalkingWithDog']	# Word 1   
        self.Xaxis = 192
        self.Yaxis = 240
        self.minFrames = 31        
        self.pathList = newpath
        self.frameIndices = []
        self.batchsize = 50
        self.k = 1
        self.current=0
        
        


 
    
    def frameLength(newpath):
        v = av.open(newpath)
        stream = next(s for s in v.streams if s.type == 'video')
        #X_data = []
        for packet in v.demux(stream):
            for frame in packet.decode():
                continue
        return frame.index
    
    def setK(self,num):
        self.k = num
    
 
    def regexBatchnum(self,path):
        re1='.*?'	# Non-greedy match on filler
        re2='g'	# Uninteresting: c
        re3='.*?'	# Non-greedy match on filler
        re4='g'	# Uninteresting: c
        re5='.*?'	# Non-greedy match on filler
        re6='g'	# Uninteresting: c
        re7='.*?'	# Non-greedy match on filler
        re8='(g)'	# Any Single Character 1        

        re9= '(' + str(self.k).zfill(2) + ')'	# Integer Number 1
        
        rg = re.compile(re1+re2+re3+re4+re5+re6+re7+re8+re9,re.IGNORECASE|re.DOTALL)
        m = rg.search(path)
        
        if(m==None):
            return False
        else:
            return True
    
    def regexClass(self,path):
        
        classnum = 0
                
        re1='.*?'	# Non-greedy match on filler
        ####################
        #self.numclasses
        i=0
        for re2 in self.classList:
            i=i+1
            rg = re.compile(re1+re2,re.IGNORECASE|re.DOTALL)
            m = rg.search(path)
            if m:
                classnum = i
                break
        return classnum
            
                
        
    
    
    def getBatch(self):
        batchCount = 0
        X = np.zeros([self.sequenceLength,self.batchsize,int(self.Xaxis/2),int(self.Yaxis/2)])
        Y = np.zeros([self.batchsize])
        
        if(self.current >= len(self.pathList)):
            self.current=0
#            print("Batches complete already")
#            return  0
        current=self.current
        for pathname in self.pathList[current:]:
            
            X_data = np.array([])
            v = av.open(pathname)
            
            self.current +=1
            
            if (self.current >= len(self.pathList)):
                self.current = 0
#                print('out of index')
#                break
            
            if(videoDataset(self.rootDir).regexBatchnum(pathname)== True):
                continue

            
            stream = next(s for s in v.streams if s.type == 'video')
            X_data = []
            for packet in v.demux(stream):
                for frame in packet.decode():
                    # some other formats gray16be, bgr24, rgb24
                    img = frame.to_nd_array(format='bgr24')
                    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    X_data.append(gray_image)
            
            X_data = np.array(X_data)
            aa= np.floor(np.linspace(1,X_data.shape[0],self.sequenceLength,endpoint = False))
            sampledX= []
            
            for i in aa:
                sampledX.append(X_data[int(i),:,:])
            sampledX=np.array(sampledX)
            
            
            #Reduced dimensions in resize_X
            resize_X = []
            
            #Resizing the (sequence_length) number of images into half size. So that the output of CNN doesn't explode 
            for p in range(sampledX.shape[0]):
                height, width = sampledX[p,:,:].shape
                gray_image = cv2.resize(sampledX[p,:,:],(int(width/2), int(height/2)), interpolation = cv2.INTER_AREA)
                resize_X.append(gray_image)
            
            resize_X = np.array(resize_X)
            
            
            
            #Now load array into the final batch array
            X[:,batchCount,:,:] = resize_X
            Y[batchCount] = int(videoDataset(self.rootDir).regexClass(pathname))
            batchCount+=1
            
            if(batchCount == self.batchsize | (batchCount < self.batchsize & self.current==len(self.pathList))):
                return X,Y