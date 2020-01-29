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
os.environ['CUDA_VISIBLE_DEVICES']='3'


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
			nn.Linear(2*64*191,32),
			nn.BatchNorm1d(32,momentum=0.01),
			nn.PReLU(32),
			nn.Dropout(0.6),
			nn.Linear(32,1),
			nn.Sigmoid()
		)
		self._initialize_weights()

		self.lstm=nn.LSTM(128,64,num_layers=1,bidirectional=True)
		self.drop3=nn.Dropout(0.3)
	def forward(self,x):
		x=x.permute(0,2,1)#N,4,201
		feature_x=self.conv1(x)#N,128,201
		feature_x=self.prelu1(feature_x.view(feature_x.size(0),-1))
		feature_x=self.drop1(feature_x)
		self.conv1_x=feature_x.view(feature_x.size(0),128,198)
		feature_x=self.conv2(self.conv1_x)#N,128,201
		feature_x=self.prelu2(feature_x.view(feature_x.size(0),-1))
		feature_x=self.drop2(feature_x)
		self.conv2_x=feature_x.view(feature_x.size(0),128,191)
		self.conv2_x=self.conv2_x.permute(2,0,1)
		feature_x,_=self.lstm(self.conv2_x)
		feature_x=feature_x.permute(1,0,2)
		feature_x=self.drop3(feature_x)
		feature_x=feature_x.contiguous().view(feature_x.size(0),-1)
		

		#feature_x=self.conv2_x.permute(0,2,1)
		#self.att_x=self.head_att(feature_x)#N,201,8
		#x=feature_x.permute(0,2,1).bmm(self.att_x)#N,128,8
		#x=x.view(x.size(0),-1)
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

	def max_norm(self):
		for m in self.modules():
			if isinstance(m,nn.Conv1d):
				norm = m.weight.norm(2, dim=None, keepdim=True)
				desired = torch.clamp(norm, 0, 3)
				param= m.weight*(desired/(1e-8+norm))
				m.weight=torch.nn.Parameter(param)

multi=1
def train():
	lr=0.002
	lambd=0.001
	MAX_EPOCHS=35
	bs=32
	fp = open('../../../../4.training_data_all_4_1/data.pkl','rb')
	data = pickle.load(fp)
	X_train_pos = data['X_train_pos']
	X_train_neg_all = data['X_train_neg']
	X_valid_pos = data['X_valid_pos']
	X_valid_neg_all =data['X_valid_neg']
	x_pos_train = X_train_pos
	train_pos_num = len(x_pos_train)
	train_neg_num = multi * train_pos_num
	train_num=train_pos_num+train_neg_num
	Y_pos_train = np.ones((train_pos_num,1),dtype=np.float32)
	Y_neg_train = np.zeros((train_neg_num,1),dtype=np.float32)
	Y_train = np.concatenate((Y_pos_train, Y_neg_train))
	x_pos_valid = X_valid_pos
	valid_pos_num = len(x_pos_valid)
	valid_neg_num = multi * valid_pos_num
	valid_num=valid_pos_num+valid_neg_num
	Y_pos_valid = np.ones((valid_pos_num,1),dtype=np.float32)
	Y_neg_valid = np.zeros((valid_neg_num,1),dtype=np.float32)
	Y_valid = np.concatenate((Y_pos_valid,Y_neg_valid))
	print ('train_sequence:', train_pos_num+train_neg_num)
	print ('valid_sequence:', valid_neg_num+valid_pos_num)
	
	for m in range(10):
		print('model {}'.format(m))
		savedir='./model_file/model_'+str(m)
		if not os.path.exists(savedir):
			os.makedirs(savedir)

		np.random.shuffle(X_train_neg_all)
		np.random.shuffle(X_valid_neg_all)
		x_neg_valid = X_valid_neg_all[:valid_neg_num]
		x_valid = np.vstack((x_pos_valid,x_neg_valid))

		net=Net()
		label_loss=nn.BCELoss(reduction='none')

		if USE_CUDA:
			net=net.cuda()
		optimizer=torch.optim.Adam(net.parameters(),lr=lr)
		scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.8)
		earlystop=EarlyStopping(patience=10)

		valid_dataset=TensorDataset(torch.from_numpy(x_valid).cuda(),torch.from_numpy(Y_valid).cuda())
		valid_loader=DataLoader(dataset=valid_dataset,batch_size=bs,shuffle=True)
		for epochs in range(MAX_EPOCHS):
			np.random.shuffle(X_train_neg_all)
			x_neg_train = X_train_neg_all[:train_neg_num]
			x_train = np.vstack((x_pos_train,x_neg_train))
			train_dataset=TensorDataset(torch.from_numpy(x_train).cuda(),torch.from_numpy(Y_train).cuda())
			train_loader=DataLoader(dataset=train_dataset,batch_size=bs,shuffle=True)
			Loss=0
			net.train()
			for i,(x,y) in enumerate(train_loader):
				x=Variable(x)
				y=Variable(y)
				bss=x.size(0)
				output=net(x)
				#loss=torch.mean(label_loss(output,y)+lambd*pterm)
				loss=torch.mean(label_loss(output,y))
				if USE_CUDA:
					Loss+=loss.cpu().data.numpy()*bss
				else:
					Loss+=loss.data.numpy()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				if i%20==0 and i!=0:
					pass
					#print('\rEpoch {}, process {}, loss {}, loss1{}, loss2{}'.format(epochs,i*bs/x_train.shape[0],Loss/i,Loss1/i,Loss2/i),end='')
			Loss/=train_num
			prob=[]
			Y=[]
			validloss=0
			net.eval()
			for i,(x,y) in enumerate(valid_loader):
				Y.append(y.cpu().data.numpy())
				x=Variable(x)
				y=Variable(y)
				bss=x.size(0)
				with torch.no_grad():
					output=net(x)
					prob.append(output.cpu().data.numpy())
					#loss=torch.mean(label_loss(output,y)+lambd*pterm)
					loss=torch.mean(label_loss(output,y))

					validloss+=loss.cpu().data.numpy()*bss
			validloss/=valid_num
			prob=np.concatenate(prob)
			Y=np.concatenate(Y)
			vfpr,vtpr,vthresholds=metrics.roc_curve(Y,prob,pos_label=1)
			vauc=metrics.auc(vfpr,vtpr)
			print('Epoch {}, trainloss {}, validloss {}, vauc: {}'.format(epochs,Loss,validloss,vauc))
			#print(' vauc: {}'.format(vauc))
			earlystop(validloss,net,savedir)
			if earlystop.early_stop:
				print('early_stopping at {}'.format(epochs))
				break
			scheduler.step()

def test():
	lambd=0.001
	fp = open('../../../../4.training_data_all_4_1/data.pkl','rb')
	data = pickle.load(fp)
	x_test_pos = data['X_test_pos']
	x_test_neg = data['X_test_neg']
	y_test_pos = np.ones((len(x_test_pos),1),dtype=np.float32)
	y_test_neg = np.zeros((len(x_test_neg),1),dtype=np.float32)

	x_test = np.vstack((x_test_pos,x_test_neg))
	y_test = np.vstack((y_test_pos,y_test_neg))
	test_dataset=TensorDataset(torch.from_numpy(x_test).cuda(),torch.from_numpy(y_test).cuda())
	test_loader=DataLoader(dataset=test_dataset,batch_size=32,shuffle=False)
	net=Net()
	net.cuda()
	label_loss=nn.BCELoss(reduction='none')
	for m in range(10):
		print('model {}'.format(m))
		modeldir='./model_file/model_'+str(m)+'/'
		modelfile=modeldir+'checkpoint.pkl'
		net.load_state_dict(torch.load(modelfile))
		net.eval()
		prob=[]
		attscore=[]
		for i,(x,y) in enumerate(test_loader):
			x=Variable(x)
			y=Variable(y)
			with torch.no_grad():
				output=net(x)
				prob.append(output.cpu().data.numpy())
				#loss=torch.mean(label_loss(output,y)+lambd*pterm)
				loss=torch.mean(label_loss(output,y))
				#attscore.append(net.att_x.cpu().data.numpy())
		prob=np.concatenate(prob)
		#attscore=np.concatenate(attscore)
		vfpr,vtpr,vthresholds=metrics.roc_curve(y_test,prob,pos_label=1)
		score=np.concatenate([y_test,prob],1)
		vauc=metrics.auc(vfpr,vtpr)
		print(vauc)
		#np.save(modeldir+'attscore.npy',attscore)
		np.save(modeldir+'score.npy',score)

if  __name__=='__main__':
    #train()
	test()

