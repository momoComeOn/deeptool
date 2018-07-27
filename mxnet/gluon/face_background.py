#-*-coding:utf-8-*-
import collections
import datetime
import gluonbook as gb
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo, nn
import numpy as np
import os
import shutil
import zipfile
import cv2
import random
def get_image(path,size):
	img = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
	#->(224,224,3) -> (3,224,224)
	img = cv2.resize(img, (size,size))
	img = np.swapaxes(img, 0, 2)
	img = np.swapaxes(img, 1, 2)
	img = img[np.newaxis, :]
	return img
	#print img.shape


def data_iter(data_label,root,size,fname):
	'''
	data_label: 名字 标签
	root: 根文件 
	size: 图片数据大小
	fname: 路径文件保存
	所有文件保存在root/名字/***下  名字与标签一对一对应
	'''
	with open(data_label,'r') as f:
		lines = f.readlines()	
		tokens = [l.strip().split(' ') for l in lines]
		idex_label = dict(((idex,label) for idex,label in tokens))
		root_ = [os.path.join(root,l) for l in idex_label.keys()]
		#label = [[int(idex_label[r.split('/')[-1]])]*len(os.listdir(r)) for r in root_]		
		#data = [get_image(os.path.join(r,j),size) for r in root_ for j in os.listdir(r)]		
	#dataiter = np.concatenate(data,axis=0)
	#dataiter = nd.array(dataiter)
	#label = nd.array(label)
	#print (dataiter.label.shape)
	#nd.save('train',dataiter)
	#return dataiter,label
		dataiter = [(os.path.join(i,j),idex_label[i.split('/')[-1]]) for i in root_ for j in os.listdir(i)]
	with open(fname,'w') as f:
		for i in dataiter:
			f.write(i[0]+' '+i[1]+'\n') 
	return dataiter


def vgg_block(num_convs,num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels,kernel_size=3,padding=1,activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2,strides=2))
    return blk

def vgg(conv_arch):
    net = nn.Sequential()
    for (num_convs,num_channels) in conv_arch:
        net.add(vgg_block(num_convs,num_channels))
    net.add(nn.Dense(4096,activation='relu'),nn.Dropout(0.5),nn.Dense(4096,activation='relu'),nn.Dropout(0.5),nn.Dense(7))
    return net

def test():
	conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512))
	net = vgg(conv_arch)
	net.initialize()
	X = nd.random.uniform(shape=(1, 1, 112, 112))
	print (X)
	for blk in net:
	    X = blk(X)
	    print(blk.name, 'output shape:\t', X.shape)



def get_data(paths,size,ctx):#读取图片数据和标签，并进行类型转换
	data = [get_image(path[0],size) for path in paths]
	label = [int(path[1]) for path in paths]
	dataiter = np.concatenate(data,axis=0)
	dataiter = nd.array(dataiter,ctx=ctx)
	label = nd.array(label,ctx=ctx)
	return dataiter,label

def get_dataiter(train_data_path,size,ctx,batch_size):#通过list的(path,label)的方式进行数据迭代
	x = len(train_data_path)
	random.shuffle(train_data_path)
#	print (train_data_path[0],train_data_path[1])
	batch = x // batch_size
	assert batch != 0
	for i in range(batch):
		begin = i*batch_size
		end = (i+1)*batch_size if (i+1)*batch_size<x else x
		data,label = get_data(train_data_path[begin:end],size,ctx)
		yield data,label

def predict(net,dataiter_path,size,ctx,batch_size):#在验证集上预测 不进行反向传播
	loss = gloss.SoftmaxCrossEntropyLoss()
	with open(dataiter_path,'r') as f:
		line = f.readlines()
		dataiter = [l.strip().split(' ') for l in line]
	x = len(dataiter)
	prev_time = datetime.datetime.now()
	train_l,train_a,count = 0.0,0.0,0
	for data,label in get_dataiter(dataiter,size,ctx,batch_size):
		outputs = net(data)
		l = loss(outputs,label)
		train_l += l.mean().asscalar()
		train_a += np.sum(np.argmax(outputs.asnumpy(),axis=1)==label.asnumpy())
		count +=  data.shape[0]
	cur_time = datetime.datetime.now()
	h, remainder = divmod((cur_time - prev_time).seconds, 3600) 
	m, s = divmod(remainder, 60)
	time_s = "time %02d:%02d:%02d" % (h, m, s)
	epoch_s = ("predict loss %f, acc %f%% "% (train_l / x, train_a/count*100))
	print(epoch_s + time_s)
	

def train(net,train_data_path,val_data_path,lr,wd,ctx,num_epoch,lr_period,lr_decay,batch_size,size):
	trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':lr,'wd':wd})
	prev_time = datetime.datetime.now()
	loss = gloss.SoftmaxCrossEntropyLoss()
	
	for epoch in range(num_epoch):
		train_l = 0.0
		train_a = 0.0
		count = 0
		if epoch >0 and epoch % lr_period == 0:
			trainer.set_learning_rate(trainer.learning_rate * lr_decay)
		with open(train_data_path,'r') as f:
			line = f.readlines()
			dataiter = [l.strip().split(' ') for l in line]
		x = len(dataiter)
		for data,label in get_dataiter(dataiter,size,ctx,batch_size):
			with autograd.record():
				outputs = net(data)
				l = loss(outputs,label)
			l.backward()
			trainer.step(batch_size)
			train_l += l.mean().asscalar()
			train_a += np.sum(np.argmax(outputs.asnumpy(),axis=1)==label.asnumpy())
			count +=  data.shape[0]
		cur_time = datetime.datetime.now()
		h, remainder = divmod((cur_time - prev_time).seconds, 3600) 
		m, s = divmod(remainder, 60)
		time_s = "time %02d:%02d:%02d" % (h, m, s)
		epoch_s = ("epoch %d, train loss %f, acc %f%% "% (epoch, train_l / x, train_a/count*100))
		print(epoch_s + time_s + ', lr ' + str(trainer.learning_rate))
		
		predict(net,val_data_path,size,ctx,batch_size) #输出预测结果

		prev_time = cur_time

#def train(net,train_data,train_data_label,lr,wd,ctx,num_epoch,lr_period,lr_decay,batch_size):
def main():
	ctx = gb.try_gpu()
	conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512))
	num_epoch = 100
	lr = 0.0001
	wd = 1e-4
	lr_period=10
	lr_decay=0.1
	net = vgg(conv_arch)
	net.initialize(ctx=ctx)
	net.hybridize()
	batch_size=128
	#print (train_iter)
	train_path = 'noface_dataiter_train.txt'
	val_path = 'noface_dataiter_val.txt'
	train(net,train_path,val_path,lr,wd,ctx,num_epoch,lr_period,lr_decay,batch_size,112)
main()
#train_iter = data_iter('./val_label.txt','/home/muyouhang/zkk/mxnet_mtcnn_face_detection/data/Val_noface_background',112,'noface_dataiter_val.txt')	
