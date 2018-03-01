import torch
from torch.autograd import Variable
import torch.nn as nn

import numpy as np


import DataLoader

path = 'C:/Users/DELL/Documents/Fourier Ptychography/Activity recognition/DVS 11/original_data/'

batchsize=50
seq_length = 5
height = 192
width = 240

#CNN parameters
learning_rate = 0.01
kH = 5
kW = 5
nInputPlane = 1
noFilters1 = 4
noFilters2 = 8
padW = (kW-1)/2
padH = (kH-1)/2
cnn_output = 5760

#RNN parameters
hidden_size = 1200
num_layers_RNN = 1
num_classes = 11




X = DataLoader.videoDataset(path)
#X_data,Y_data = X.getBatch()

#X_data = Variable(torch.FloatTensor(X_data), requires_grad=False)
#Y_data = Variable(torch.Tensor(Y_onehot), requires_grad=False)

class CNN(nn.Module):
    def __init__(self, noFilters1, noFilters2, kH, width):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, noFilters1, kernel_size= kH, padding= int((kH-1)/2)),
            nn.BatchNorm2d(noFilters1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(noFilters1, noFilters2, kernel_size=kH, padding= int((kH-1)/2)),
            nn.BatchNorm2d(noFilters2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        #self.fc = nn.Linear( int((width)/4 * (width)/4 *noFilters2), 11)
        #softmax
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        #out = self.fc(out)
        return out
        
cnn = CNN(noFilters1, noFilters2, kH, width)


# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.soft = nn.Softmax()
    
    def forward(self, x):
        # Set initial states 
        h0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size)) 
        c0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))
        
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))  
        
        # Decode hidden state of last time step
        out = self.fc(out)  
        out = self.soft(out)
        return out

rnn = RNN(cnn_output, hidden_size, num_layers_RNN, num_classes)



for i in range(320):
    
    X_data,Y_data = X.getBatch()
    Y_onehot = (np.arange(num_classes) == Y_data[:,None]).astype(np.float32)
    
    X_data = Variable(torch.FloatTensor(X_data))
    Y_data = Variable(torch.Tensor(Y_onehot), requires_grad=False)
    
    RNNinput = np.zeros([seq_length,batchsize,cnn_output])
    RNNinput = Variable(torch.from_numpy(RNNinput).float())
    
    RNNoutput = np.zeros([seq_length,batchsize,num_classes])
    RNNoutput = Variable(torch.from_numpy(RNNoutput).float())
    
    T = []
    temp = []
    for i in range(seq_length):
        T.append(X_data[i,:,:,:].unsqueeze(1))
    
    for t in T:
        temp.append(cnn(t))
    
    for k in range(batchsize):
        for m in range(len(temp)):
            TEMP = temp[m]
            RNNinput[m,k,:]= TEMP[k,:]
            
    Y_out = rnn(RNNinput)

A1=RNNinput[0,:,:] ;RNNoutput[0,:,:] = rnn(A1);     
A2=RNNinput[1,:,:] ;RNNoutput[1,:,:] = rnn(A2)
A3=RNNinput[2,:,:] ;RNNoutput[2,:,:] = rnn(A3)
A4=RNNinput[3,:,:] ;RNNoutput[3,:,:] = rnn(A4) 
A5=RNNinput[4,:,:] ;RNNoutput[4,:,:] = rnn(A5)       

  


 
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)


cnn(x1)
cnn(X_data.permute(1,0,2,3))


'''
x = Variable(torch.randn(seq_length, , width, height))
y = Variable(torch.randn(N))

model = nn.Sequential()
model:torch.add( nn.VolumetricConvolution(1, noFilters1, kT, kH, 1, 1, 1, padT, padW, padH ))  
model:torch.add( torch.nn.Tanh())
model:torch.add( torch.nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2 ))

model:torch.add( torch.nn.VolumetricConvolution(noFilters1, noFilters2, kT, kW, kH, 1, 1, 1, padT, padW, padH ))
model:torch.add( torch.nn.Tanh())
model:torch.add( torch.nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2 ))
'''