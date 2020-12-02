import torch
import torch.nn as nn

class seekndestroy(nn.Module):

    def __init__(self, input_dim=30, output_dim=2):
        super().__init__()

        #deep
        self.Conv1 = nn.Conv1d(input_dim, 14, 5, stride = 5, padding =5)
        self.Pool1 = nn.MaxPool1d(2)
        self.deep_1 = nn.Linear(31, 128, bias=True)
        self.deep_2 = nn.Linear(128, 30, bias=True)
        
        self.LSTM = nn.LSTM(input_size=32,
                         hidden_size=50,
                         num_layers=1,
                         bidirectional=False)
        # Output layer 
        self.l_out = nn.Linear(in_features=50,
                            out_features=32,
                            bias=False)
        
        
        self.deep_3 = nn.Linear(32, 32, bias=True)
        self.deep_4 = nn.Linear(32, output_dim, bias=True)
        
        self.softmax = nn.LogSoftmax(dim=1)

        #conat
       
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        
        # Convolution Block
        print(inputs.shape)
        Sensor_Reading = inputs[:,:30]
        Input_Args = inputs[:,30:]   ## [dist/maxdist ratio to goal, angle, sign, POSITION, VELOCITY]
        D_and_A_and_S = input_Args[:,:3]
        P_and_V = Input_Args[:,-2:]  ## Append P and V to output of convolution 
        xd = Sensor_Reading.unsqueeze(0).permute(0, 2, 1)
        print(xd.shape)
        xd = self.Conv1(xd)
        xd = self.relu(xd)
#         xd = self.Pool1(xd)

        # Fully Connected Block
        print(xd.shape)
        xd = xd.view(1,xd.shape[1]*xd.shape[2])
        # print(xd.shape)
        # print(xd.shape)
        xd = torch.cat((xd,D_and_A_and_S), 1)
        xd = self.deep_1(xd)
        xd = self.relu(xd)
        xd = self.deep_2(xd)
        xd = self.relu(xd)
        
        xd = torch.cat((xd,P_and_V), 1)
        xd = xd.unsqueeze(0)
          # LSTM Block
        # RNN returns output and last hidden state
        x, (h, c) = self.LSTM(xd)
        
        # Flatten output for feed-forward layer
        x = x.view(-1, self.LSTM.hidden_size)
        
        # Output layer
        xd = self.l_out(x)
        
        xd = self.deep_3(xd)
        xd = self.relu(xd)
        xd = self.deep_4(xd)
        xd = self.relu(xd)
        
      
        
#         x = self.softmax(x)
        x = self.sigmoid(xd)

        return x