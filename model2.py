{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch.nn as nn\n",
    "\n",
    "class seekndestroy(nn.Module):\n",
    "\n",
    "    def __init__(self, input_dim=35, output_dim=2):\n",
    "        super().__init__()\n",
    "\n",
    "        #deep\n",
    "        self.Conv1 = nn.Conv1d(input_dim[:,:30], 7, 5, stride = 5, padding =5)\n",
    "        self.Pool1 = nn.MaxPool1d(2)\n",
    "        self.deep_1 = nn.Linear(9, 128, bias=True)\n",
    "        self.deep_2 = nn.Linear(128, 64, bias=True)\n",
    "        self.deep_3 = nn.Linear(64, 32, bias=True)\n",
    "        self.deep_4 = nn.Linear(32, 16, bias=True)\n",
    "        self.LSTM = nn.LSTM(input_size=16,\n",
    "                         hidden_size=50,\n",
    "                         num_layers=1,\n",
    "                         bidirectional=False)\n",
    "        # Output layer \n",
    "        self.l_out = nn.Linear(in_features=50,\n",
    "                            out_features=output_dim,\n",
    "                            bias=False)\n",
    "        \n",
    "        self.softmax = nn.LogSoftmax(dim=1)\n",
    "\n",
    "        #conat\n",
    "       \n",
    "        self.relu = nn.ReLU()\n",
    "        self.tanh = nn.Tanh()\n",
    "        self.sigmoid = nn.Sigmoid()\n",
    "        \n",
    "    def forward(self, inputs):\n",
    "        \n",
    "        # Convolution Block\n",
    "        Sensor_Reading = inputs[:,:30]\n",
    "        Input_Args = inputs[:,30:]   ## [dist/maxdist ratio to goal, angle, sign, POSITION, VELOCITY]\n",
    "        P_and_V = Input_Args[:,-2:]  ## Append P and V to output of convolution \n",
    "        xd = Sensor_Reading.unsqueeze(0).permute(0, 2, 1)\n",
    "        xd = self.Conv1(xd)\n",
    "        xd = self.relu(xd)\n",
    "#         xd = self.Pool1(xd)\n",
    "\n",
    "        # Fully Connected Block\n",
    "        xd = xd.view(1,7)\n",
    "        xd = xd.append(P_and_V)\n",
    "        xd = self.deep_1(xd)\n",
    "        xd = self.relu(xd)\n",
    "        xd = self.deep_2(xd)\n",
    "        xd = self.relu(xd)\n",
    "        xd = self.deep_3(xd)\n",
    "        xd = self.relu(xd)\n",
    "        xd = self.deep_4(xd)\n",
    "        xd = self.relu(xd)\n",
    "        \n",
    "        # LSTM Block\n",
    "        # RNN returns output and last hidden state\n",
    "        x, (h, c) = self.lstm(xd)\n",
    "        \n",
    "        # Flatten output for feed-forward layer\n",
    "        x = x.view(-1, self.lstm.hidden_size)\n",
    "        \n",
    "        # Output layer\n",
    "        x = self.l_out(x)\n",
    "        \n",
    "        x = self.softmax(x)\n",
    "\n",
    "        return x\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
