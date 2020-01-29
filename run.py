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
from model import Net
import argparse
USE_CUDA=True
def train(datafile,lr,bs,multi,savedir):
	MAX_EPOCHS=35
	fp = open(datafile,'rb')
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

def test(datafile,modeldir):
	fp = open(datafile,'rb')
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
			loss=torch.mean(label_loss(output,y))
			attscore.append(net.att_x.cpu().data.numpy())
	prob=np.concatenate(prob)
	attscore=np.concatenate(attscore)
	vfpr,vtpr,vthresholds=metrics.roc_curve(y_test,prob,pos_label=1)
	score=np.concatenate([y_test,prob],1)
	vauc=metrics.auc(vfpr,vtpr)
	print(vauc)
	np.save(modeldir+'attscore.npy',attscore)
	np.save(modeldir+'score.npy',score)

if  __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-m','--mode',default='train',help='train or test mode')
	parser.add_argument('-d','--data',help='datafile for train and test')
	parser.add_argument('-s','--save_dir',default='./model_output/',help='directory of model output')
	parser.add_argument('-r','--ratio',default=1,type=int,help='ratio of negative to positive samples')
	parser.add_argument('-lr','--learning_rate',default=0.02,type=float,help='learning rate')
	parser.add_argument('-bs','--batch_size',default=32,type=float,help='batch_size')
	parser.add_argument('-md','--model_dir',help='model saving directory for testing')
	args=parser.parse_args()
	if args.mode=='train':
		print('training mode...')
		if args.data:
			train(args.data,args.learning_rate,args.batch_size,args.ratio,args.save_dir)
		else:
			print('Please specify datafile')
	if args.mode=='test' and args.model_dir:
		print('testing mode...')
		if args.data:
			test(args.data,args.model_dir)
		else:
			print('Please specify datafile')
	else:
		print('Please input correct mode')

