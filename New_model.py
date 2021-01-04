import torch
import torch.nn as nn
import numpy as np

class seekndestroy(nn.Module):

    def __init__(self, input_dim=35, output_dim=2):
        super().__init__()

        
        self.deep_1 = nn.Linear(35, 35, bias=True)
        self.deep_2 = nn.Linear(50, output_dim, bias=True)
        
        self.LSTM = nn.LSTM(input_size=35,
                         hidden_size=50,
                         num_layers=1,
                         bidirectional=False)
        # Output layer 
        self.l_out = nn.Linear(in_features=50,
                            out_features=32,
                            bias=False)
        
        
#         self.BN = nn.BatchNorm1d(128) ##, affine=False)
        
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
          x = self.deep_1(inputs)
          x = self.relu(x)

          x, (h, c) = self.LSTM(x)
          x = x.view(-1, self.LSTM.hidden_size)
          x = self.relu(x)

          x = self.deep_2(x)
          x = self.tanh(x)

          return x