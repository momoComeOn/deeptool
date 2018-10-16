
#-*-coding:utf-8-*-

def lstm():
    with open('./song.txt') as f:
        corpus_chars = f.read()
    print(corpus_chars[0:49])

    print len(corpus_chars)
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    print len(corpus_chars)

    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])

    vocab_size = len(char_to_idx)

    print('vocab size:', vocab_size)



    corpus_indices = [char_to_idx[char] for char in corpus_chars]

    sample = corpus_indices[:40]

    print('chars: \n', ''.join([idx_to_char[idx] for idx in sample]))
    print('\nindices: \n', sample)


    import random
    from mxnet import nd

    def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
        # 减一是因为label的索引是相应data的索引加一
        num_examples = (len(corpus_indices) - 1) // num_steps
        epoch_size = num_examples // batch_size
        # 随机化样本
        example_indices = list(range(num_examples))
        random.shuffle(example_indices)

        # 返回num_steps个数据
        def _data(pos):
            return corpus_indices[pos: pos + num_steps]

        for i in range(epoch_size):
            # 每次读取batch_size个随机样本
            i = i * batch_size
            batch_indices = example_indices[i: i + batch_size]
            data = nd.array(
                [_data(j * num_steps) for j in batch_indices], ctx=ctx)
            label = nd.array(
                [_data(j * num_steps + 1) for j in batch_indices], ctx=ctx)
            yield data, label
            

    def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
        corpus_indices = nd.array(corpus_indices, ctx=ctx)
        data_len = len(corpus_indices)
        batch_len = data_len // batch_size

        indices = corpus_indices[0: batch_size * batch_len].reshape((
            batch_size, batch_len))
        # 减一是因为label的索引是相应data的索引加一
        epoch_size = (batch_len - 1) // num_steps

        for i in range(epoch_size):
            i = i * num_steps
            data = indices[:, i: i + num_steps]
            label = indices[:, i + 1: i + num_steps + 1]
            yield data, label
            
    my_seq = list(range(30))

    for data, label in data_iter_random(my_seq, batch_size=2, num_steps=3):
        print 'data: ', data, '\nlabel:', label, '\n'


    print nd.one_hot(nd.array([0, 2]), vocab_size)

    def get_inputs(data):
        return [nd.one_hot(X, vocab_size) for X in data.T]

    inputs = get_inputs(data)
    print data.T

    print('input length: ', len(inputs))
    print('input[0] shape: ', inputs[0].shape)

    import mxnet as mx

    # 尝试使用GPU
    import sys
    sys.path.append('..')
    import gluonbook as gb
    ctx = gb.try_gpu()
    print('Will use', ctx)

    input_dim = vocab_size
    # 隐含状态长度
    hidden_dim = 256
    output_dim = vocab_size
    std = .01

    def get_params_rnn():
        # 隐含层
        W_xh = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
        W_hh = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
        b_h = nd.zeros(hidden_dim, ctx=ctx)

        # 输出层
        W_hy = nd.random_normal(scale=std, shape=(hidden_dim, output_dim), ctx=ctx)
        b_y = nd.zeros(output_dim, ctx=ctx)

        params = [W_xh, W_hh, b_h, W_hy, b_y]
        for param in params:
            param.attach_grad()
        return params

        
    def rnn(inputs, state, *params):
        # inputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵。
        # H: 尺寸为 batch_size * hidden_dim 矩阵。
        # outputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵。
        H = state
        W_xh, W_hh, b_h, W_hy, b_y = params
        outputs = []
        for X in inputs:
            H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
            Y = nd.dot(H, W_hy) + b_y
            outputs.append(Y)
        return (outputs, H)

    def get_params():
        # 输入门参数
        W_xi = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
        W_hi = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
        b_i = nd.zeros(hidden_dim, ctx=ctx)

        # 遗忘门参数
        W_xf = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
        W_hf = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
        b_f = nd.zeros(hidden_dim, ctx=ctx)

        # 输出门参数
        W_xo = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
        W_ho = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
        b_o = nd.zeros(hidden_dim, ctx=ctx)

        # 候选细胞参数
        W_xc = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
        W_hc = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
        b_c = nd.zeros(hidden_dim, ctx=ctx)

        # 输出层
        W_hy = nd.random_normal(scale=std, shape=(hidden_dim, output_dim), ctx=ctx)
        b_y = nd.zeros(output_dim, ctx=ctx)

        params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
                  b_c, W_hy, b_y]
        for param in params:
            param.attach_grad()
        return params
      
    def lstm_rnn(inputs, state_h, state_c, *params):
        # inputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵
        # H: 尺寸为 batch_size * hidden_dim 矩阵
        # outputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
         W_hy, b_y] = params

        H = state_h
        C = state_c
        outputs = []
        for X in inputs:
            I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
            F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
            O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
            C_tilda = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)
            C = F * C + I * C_tilda
            H = O * nd.tanh(C)
            Y = nd.dot(H, W_hy) + b_y
            outputs.append(Y)
        return (outputs, H, C)
      
    state = nd.zeros(shape=(data.shape[0], hidden_dim), ctx=ctx)

    params = get_params()
    #outputs, state_new = rnn(get_inputs(data.as_in_context(ctx)), state, *params)

    #print('output length: ',len(outputs))
    #print('output[0] shape: ', outputs[0].shape)
    #print('state shape: ', state_new.shape)


    def predict_rnn(rnn, prefix, num_chars, params, hidden_dim, ctx, idx_to_char,
                    char_to_idx, get_inputs, is_lstm=False):
        # 预测以 prefix 开始的接下来的 num_chars 个字符。
        prefix = prefix.lower()
        state_h = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
        if is_lstm:
            # 当RNN使用LSTM时才会用到，这里可以忽略。
            state_c = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
        output = [char_to_idx[prefix[0]]]
        for i in range(num_chars + len(prefix)):
            X = nd.array([output[-1]], ctx=ctx)
            # 在序列中循环迭代隐含变量。
            if is_lstm:
                # 当RNN使用LSTM时才会用到，这里可以忽略。
                Y, state_h, state_c = rnn(get_inputs(X), state_h, state_c, *params)
            else:
                Y, state_h = rnn(get_inputs(X), state_h, *params)
            if i < len(prefix)-1:
                next_input = char_to_idx[prefix[i+1]]
            else:
                next_input = int(Y[0].argmax(axis=1).asscalar())
            output.append(next_input)
        return ''.join([idx_to_char[i] for i in output])
        
    def grad_clipping(params, theta, ctx):
        if theta is not None:
            norm = nd.array([0.0], ctx)
            for p in params:
                norm += nd.sum(p.grad ** 2)
            norm = nd.sqrt(norm).asscalar()
            if norm > theta:
                for p in params:
                    p.grad[:] *= theta / norm
                    
    from mxnet import autograd
    from mxnet import gluon
    from math import exp

    def train_and_predict_rnn(rnn, is_random_iter, epochs, num_steps, hidden_dim,
                              learning_rate, clipping_theta, batch_size,
                              pred_period, pred_len, seqs, get_params, get_inputs,
                              ctx, corpus_indices, idx_to_char, char_to_idx,
                              is_lstm=False):
        if is_random_iter:
            data_iter = data_iter_random
        else:
            data_iter = data_iter_consecutive
        params = get_params()

        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        for e in range(1, epochs + 1):
            # 如使用相邻批量采样，在同一个epoch中，隐含变量只需要在该epoch开始的时候初始化。
            if not is_random_iter:
                state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
                if is_lstm:
                    # 当RNN使用LSTM时才会用到，这里可以忽略。
                    state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            train_loss, num_examples = 0, 0
            for data, label in data_iter(corpus_indices, batch_size, num_steps,
                                         ctx):
                # 如使用随机批量采样，处理每个随机小批量前都需要初始化隐含变量。
                if is_random_iter:
                    state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
                    if is_lstm:
                        # 当RNN使用LSTM时才会用到，这里可以忽略。
                        state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
                with autograd.record():
                    # outputs 尺寸：(batch_size, vocab_size)
                    if is_lstm:
                        # 当RNN使用LSTM时才会用到，这里可以忽略。
                        outputs, state_h, state_c = rnn(get_inputs(data), state_h,
                                                        state_c, *params)
                    else:
                        outputs, state_h = rnn(get_inputs(data), state_h, *params)
                    # 设t_ib_j为i时间批量中的j元素:
                    # label 尺寸：（batch_size * num_steps）
                    # label = [t_0b_0, t_0b_1, ..., t_1b_0, t_1b_1, ..., ]
                    label = label.T.reshape((-1,))
                    # 拼接outputs，尺寸：(batch_size * num_steps, vocab_size)。
                    outputs = nd.concat(*outputs, dim=0)
                    # 经上述操作，outputs和label已对齐。
                    loss = softmax_cross_entropy(outputs, label)
                loss.backward()

                grad_clipping(params, clipping_theta, ctx)
                gb.SGD(params, learning_rate)

                train_loss += nd.sum(loss).asscalar()
                num_examples += loss.size
                print train_loss
                print num_examples
            if e % pred_period == 0:
                print num_examples
                print train_loss
                print("Epoch %d. Perplexity %f" % (e,
                                                   exp(train_loss/num_examples)))
                for seq in seqs:
                    print(' - ', predict_rnn(rnn, seq, pred_len, params,
                          hidden_dim, ctx, idx_to_char, char_to_idx, get_inputs,
                          is_lstm))
                print()
                

                
                
                
    epochs = 200
    num_steps = 15
    learning_rate = 0.1
    batch_size = 15

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    seq1 = '胚勾'
    seq2 = '勾勒'
    seq3 = '瓶身描绘'
    seqs = [seq1, seq2, seq3]

    for data, label in data_iter_consecutive(corpus_indices, 15, 15,ctx):
        print data
        print label

    #train_and_predict_rnn(rnn=lstm_rnn, is_random_iter=False, epochs=200, num_steps=num_steps,
                         # hidden_dim=hidden_dim, learning_rate=0.2,
                         # clipping_theta=5, batch_size=batch_size, pred_period=20,
                         # pred_len=100, seqs=seqs, get_params=get_params,
                         # get_inputs=get_inputs, ctx=ctx,
                         # corpus_indices=corpus_indices, idx_to_char=idx_to_char,
                         # char_to_idx=char_to_idx)
    gb.train_and_predict_rnn(rnn=lstm_rnn, is_random_iter=False, epochs=200,
                             num_steps=15, hidden_dim=hidden_dim,
                             learning_rate=0.2, clipping_norm=5,
                             batch_size=15, pred_period=20, pred_len=100,
                             seqs=seqs, get_params=get_params,
                             get_inputs=get_inputs, ctx=ctx,
                             corpus_indices=corpus_indices,
                             idx_to_char=idx_to_char, char_to_idx=char_to_idx,
                             is_lstm=True)
         

import numpy as np
import os
from mxnet import nd  
import mxnet as mx

# 尝试使用GPU
import sys
sys.path.append('..')
import gluonbook as gb
ctx = mx.gpu(0)
print ctx
from mxnet import autograd
from mxnet import gluon
from math import exp
 
train_path = './train'
val_path = './val'
train_data_path = './train/train_label.txt'
val_data_path = './val/val_label.txt'

std = .01
input_dim = 20
hidden_dim = 256
vocab_size = 20
output_dim = 7
batch_size = 32

def get_params():
    # 输入门参数
    W_xi = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hi = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_i = nd.zeros(hidden_dim, ctx=ctx)

    # 遗忘门参数
    W_xf = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hf = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_f = nd.zeros(hidden_dim, ctx=ctx)

    # 输出门参数
    W_xo = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_ho = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_o = nd.zeros(hidden_dim, ctx=ctx)

    # 候选细胞参数
    W_xc = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hc = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_c = nd.zeros(hidden_dim, ctx=ctx)

    # 输出层
    W_hy = nd.random_normal(scale=std, shape=(hidden_dim, output_dim), ctx=ctx)
    b_y = nd.zeros(output_dim, ctx=ctx)

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params
      
def lstm_rnn(inputs, state_h, state_c, *params):
    # inputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵
    # H: 尺寸为 batch_size * hidden_dim 矩阵
    # outputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hy, b_y] = params

    H = state_h
    C = state_c
    outputs = []
    for X in inputs:
        I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
        F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
        O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
        C_tilda = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * nd.tanh(C)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return (outputs, H, C)

        
        
        

def data(path,train=True):
    files = file(path)
    for line in files.readlines():
        li = line.strip().split(' ')
        if train:
            data_ori = np.load(os.path.join(train_path,li[0])+'.npy')
            yield data_ori,li[1]
        else :
            data_ori = np.load(os.path.join(val_path,li[0])+'.npy')
            yield data_ori,li[1]

def data_iter_consecutive(data_ori, batch_size =1, num_steps=16, ctx=None):
    corpus_indices = nd.array(data_ori, ctx=ctx)
    data_len = corpus_indices.shape[1]
    batch_len = corpus_indices.shape[0]
    data_len = data_len // num_steps
    for i in range(data_len):
        i = i * num_steps
        yield corpus_indices[:,i:i+num_steps]

def get_input(data_iter,ctx):
	return [nd.array(x,ctx=ctx) for x in data_iter]

def grad_clipping(params, theta, ctx):
    if theta is not None:
        norm = nd.array([0.0], ctx)
        for param in params:
            norm += (param.grad ** 2).sum()
        norm = norm.sqrt().asscalar()
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm
from scipy.stats import mode
            
def train_and_predict_rnn(rnn, epochs, num_steps, hidden_dim,learning_rate,pred_period, pred_len,
                             get_params,ctx,is_lstm=True,batch_size=1):
    #params = get_params()
    params = nd.load('params_best.txt')
    for param in params:
        param.attach_grad()
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
	
    for e in range(1, epochs + 1):
        
        if e % 20 == 1:
            count = 0
            print '---*---'    
            #print train_loss
            #print num_examples
	    batch = 1
   	    train_loss = 0
	    num_examples = 0
            for data_iter, label in data_batch(batch,val_data_path,train=False):
		state_h = nd.zeros(shape=(batch, hidden_dim), ctx=ctx)
                state_c = nd.zeros(shape=(batch, hidden_dim), ctx=ctx)
                result = np.zeros((7,))
                outputs, state_h, state_c = rnn(get_input(data_iter,ctx), state_h, state_c, *params)
                outputs = nd.concat(*outputs, dim=0)
		h = outputs.asnumpy()	
		#result = h[-1]+h[-2]+h[-3]+h[-4]+h[-5]
		result = np.mean(h,axis=0)
		#result = np.mean(h,axis=1)
		#x = int(mode(result).mode)
		if (int(label[0]) == np.argmax(result)):
		#if (int(label[0]) == x):
                    count +=1
	    	label = nd.array(label,ctx=ctx)
                loss = softmax_cross_entropy(outputs, label)
	    
            train_loss += nd.sum(loss).asscalar()
            num_examples += loss.size
            print count
	    print train_loss/num_examples 
        train_loss, num_examples = 0, 0
        for data_iter, label_iter in data_batch(batch_size,train_data_path,True):
            step,batch,_ = data_iter.shape
	    state_h = nd.zeros(shape=(batch, hidden_dim), ctx=ctx)
            state_c = nd.zeros(shape=(batch, hidden_dim), ctx=ctx)
	    #step = 16
            for data,label in data_step(data_iter,label_iter,step,batch):
	    	#state_h = nd.zeros(shape=(batch, hidden_dim), ctx=ctx)
            	#state_c = nd.zeros(shape=(batch, hidden_dim), ctx=ctx)
	    	with autograd.record():
                	# outputs 尺寸：(batch_size, vocab_size)
                	if is_lstm:
                    	# 当RNN使用LSTM时才会用到，这里可以忽略。
                    		data_input = get_input(data,ctx)
		    		outputs, state_h, state_c = rnn(data_input, state_h,
                                                        state_c, *params)
                		# 设t_ib_j为i时间批量中的j元素:
                		# label 尺寸：（batch_size * num_steps）
                		# label = [t_0b_0, t_0b_1, ..., t_1b_0, t_1b_1, ..., ]
                    
                		#label = label.T.reshape((-1,))
                		# 拼接outputs，尺寸：(batch_size * num_steps, vocab_size)。
                	outputs = nd.concat(*outputs, dim=0)
                	# 经上述操作，outputs和label已对齐。
	    		label = nd.array(label,ctx=ctx)
                	loss = softmax_cross_entropy(outputs, label)
            	loss.backward()
	    	grad_clipping(params, 5 , ctx)	
            	gb.SGD(params, learning_rate)
           	train_loss += nd.sum(loss).asscalar()
            	num_examples += loss.size
        print '---*---'    
	print train_loss
	print num_examples
        print train_loss/num_examples
        #print num_examples    
    mx.nd.save('./params_best.txt',params)
        
                
                
                
def paixu(path,train=True):
        idex,data_ori,label_ori=[],[],[]
        for data_iter,label in data(path,train):
                idex.append(data_iter.shape[1])
                data_ori.append(data_iter)
		label_ori.append(label)
        label = np.array(label_ori)
        idex = np.array(idex)
        #print idex.shape
        idex_idex = np.argsort(idex)
        #print idex_idex.shape
        return [data_ori[i] for i in idex_idex],[label[i] for i in idex_idex]
        

def data_batch(batch_size,path,train):
	data ,label= paixu(path,train)
	count = len(data)/batch_size
	if train :
		for i in range(count+1):
			i = i*batch_size
			end = i + batch_size
			if end > len(data):
				end = len(data)
			x,y = data[i].shape
			result = np.empty(shape=[y,0,20],dtype=float)
			for j in range(i,end):
				result_temp = data[j].T[0:y,:].reshape(y,1,20)
				result = np.append(result,result_temp,axis=1)
			yield result,label[i:end]*y
	else :
		for i in range(count):
			i = i*batch_size
			end = i + batch_size
			if end > len(data):
				end = len(data)
			x,y = data[i].shape
			result = np.empty(shape=[y,0,20],dtype=float)
			for j in range(i,end):
				result_temp = data[j].T[0:y,:].reshape(y,1,20)
				result = np.append(result,result_temp,axis=1)
			yield result,label[i:end]*y
def data_step(data,label,step,batch):
	x,_,_=data.shape
	count = (x-1) / step
	for i in range(count+1):
		i = i * step
		end = i + step
		if end > x:
			end = x
		yield data[i:end],label[i*batch:end*batch]
lr = 0.001
train_and_predict_rnn(rnn=lstm_rnn, epochs=100, num_steps=16, hidden_dim=hidden_dim,learning_rate=lr, batch_size=batch_size, pred_period=20, pred_len=100,get_params=get_params,ctx=ctx,is_lstm=True)
