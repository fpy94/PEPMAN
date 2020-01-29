import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader,TensorDataset
from early_stopping import EarlyStopping
import os
import pickle
from torch.autograd import Variable, Function
#os.environ['CUDA_VISIBLE_DEVICES']='2'


USE_CUDA=True
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.conv1=nn.Sequential(
			nn.Conv1d(4,128,4),
			nn.BatchNorm1d(128,momentum=0.01)
		)
		self.prelu1=nn.PReLU(128*198)
		self.drop1=nn.Dropout(0.8)
		self.conv2=nn.Sequential(
			nn.Conv1d(128,128,8),
			nn.BatchNorm1d(128,momentum=0.01)
		)
		self.prelu2=nn.PReLU(128*191)
		self.drop2=nn.Dropout(0.8)
		self.head_att=nn.Sequential(
			nn.Linear(128,128),
			nn.Tanh(),
			nn.Linear(128,1),
			nn.Softmax(dim=1)
		)

		self.MLP=nn.Sequential(
			nn.Linear(128*191,32),
			nn.BatchNorm1d(32,momentum=0.01),
			nn.PReLU(32),
			nn.Dropout(0.4),
			nn.Linear(32,1),
			nn.Sigmoid()
		)
		self._initialize_weights()

	def forward(self,x):
		x=x.permute(0,2,1)#N,4,201
		feature_x=self.conv1(x)#N,128,201
		feature_x=self.prelu1(feature_x.view(feature_x.size(0),-1))
		feature_x=self.drop1(feature_x)
		self.conv1_x=feature_x.view(feature_x.size(0),128,198)
		feature_x=self.conv2(self.conv1_x)#N,128,201
		feature_x=self.prelu2(feature_x.view(feature_x.size(0),-1))
		feature_x=self.drop2(feature_x)
		x=self.MLP(feature_x)

		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m,nn.Conv1d):
				nn.init.xavier_uniform_(m.weight,gain=1)
				nn.init.zeros_(m.bias)
			if isinstance(m,nn.Linear):
				nn.init.xavier_uniform_(m.weight,gain=1)
				nn.init.zeros_(m.bias)


