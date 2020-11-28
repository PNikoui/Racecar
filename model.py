import torch.nn as nn

class seekndestroy(nn.Module):

    def __init__(self, input_dim=35, output_dim=2):
        super().__init__()

        #deep
        self.deep_1 = nn.Linear(input_dim, 128, bias=True)
        self.Conv1 = nn.Conv1d(128, 64, 3, stride = 3, padding =3)
        self.Pool1 = nn.MaxPool1d(2)
#         self.deep_2 = nn.Linear(128, 64, bias=True)
        self.deep_3 = nn.Linear(64, 32, bias=True)
        self.deep_4 = nn.Linear(32, 16, bias=True)

        #conat
        self.out = nn.Linear(16, output_dim, bias=True)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):

        xd = self.deep_1(inputs)
        xd = self.relu(xd)
        xd = xd.unsqueeze(0).permute(0, 2, 1)
#         xd = self.deep_2(xd)
        xd = self.Conv1(xd)
        xd = self.Pool1(xd)
        xd = xd.view(1,64)
        xd = self.relu(xd)
        xd = self.deep_3(xd)
        xd = self.relu(xd)
        xd = self.deep_4(xd)
        xd = self.relu(xd)

        m = self.out(xd)
#         m = self.tanh(m)
        m = self.sigmoid(m)

        return m
