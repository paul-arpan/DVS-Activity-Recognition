import torch
import os
import numpy as np
import re

import cv2
import av
path = 'C:/Users/DELL/Documents/Fourier Ptychography/Activity recognition/DVS 11/original_data/'

def imgstck(newpath):
    v = av.open(newpath)
    stream = next(s for s in v.streams if s.type == 'video')
    #X_data = []
    for packet in v.demux(stream):
        for frame in packet.decode():
            continue
#    return frame.index
        
        # some other formats gray16be, bgr24, rgb24
    img = frame.to_nd_array(format='bgr24')
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_data.append(gray_image)

    X_data = np.array(X_data)
    
    return X_data


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

#newpath = path + name[1]+'/' + file[1][2]
Baski = np.array([])
Min = np.array([])
#BasketBall

for path in newpath:
    Baski = np.append(Baski,imgstck(newpath))
    
#    Min = np.append(Min,min(Baski))         Baski = np.append(Baski,imgstck(newpath))

 

#MinFrames = min(Min)



'''
re1='.*?'	# Non-greedy match on filler
re2='g'	# Uninteresting: c
re3='.*?'	# Non-greedy match on filler
re4='g'	# Uninteresting: c
re5='.*?'	# Non-greedy match on filler
re6='g'	# Uninteresting: c
re7='.*?'	# Non-greedy match on filler
re8='(g)'	# Any Single Character 1
re9='(02)'	# Integer Number 1

rg = re.compile(re1+re2+re3+re4+re5+re6+re7+re8+re9,re.IGNORECASE|re.DOTALL)
m = rg.search(newpath)
if m:
    c1=m.group(1)
    int1=m.group(2)
    print "("+c1+")"+"("+int1+")"+"\n"
    
    
y_onehot = y.numpy()
y_onehot = (np.arange(num_classes) == Y_data[:,None]).astype(np.float32)
y_onehot = torch.from_numpy(y_onehot)    
'''

classnum = 0
        
re1='.*?'	# Non-greedy match on filler
reList=['Basketball','Biking','Diving','GolfSwing','HorseRiding','SoccerJuggling','Swing','TennisSwing','TrampolineJumping','VolleyballSpiking','WalkingWithDog']	# Word 1   
i=0
for re2 in reList:
    i=i+1
    rg = re.compile(re1+re2,re.IGNORECASE|re.DOTALL)
    m = rg.search(newpath[1000])
    if m:
        classnum = i
        break
print(classnum)
    
    