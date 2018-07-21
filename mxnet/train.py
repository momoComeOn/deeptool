import mxnet as mx
import os
import numpy as np
import logging
import parser
import symbol
import argparse

def sym_arg_aux(flag,output=6,prefix='segnet_new',epoch=2):
	if flag == 'restart':
		sym = symbol.sym(output=output)
		return sym,{},{}
	if flag == 'fineturn':
		import load_model
		sym = symbol.sym(output=output)
		arg = load_model.mxnet_test('VGG_FC_ILSVRC_16_layers',74).arg_aux()
		arg_ = load_model.mxnet_test('VGG_FC_ILSVRC_16_layers',74).arg_aux_1()
		arg.update(arg_)
		return sym,arg,{}
	if flag == 'continue':
		return mx.model.load_checkpoint(prefix,epoch)
	sym = symbol.sym(output=output)
	return sym,{},{}

def parse_():	
    parse = argparse.ArgumentParser(description='train model!')
    parse.add_argument('--flag',help='how to bulid model',default='fineturn',type=str,choices=['restart', 'fineturn', 'continue'])
    parse.add_argument('--output',help='number of output',default=6,type=int)
    parse.add_argument('--continue_prefix',help='how to build model',default='./segnet_new',type=str)
    parse.add_argument('--continue_epoch',help='how to build model',default=2,type=int) 
    parse.add_argument('--train_path',help='path',default='./train.npy',type=str)
    parse.add_argument('--train_label_path',help='path',default='./train_label.npy',type=str)
    parse.add_argument('--val_path',help='path',default='./val.npy',type=str)
    parse.add_argument('--val_label_path',help='path',default='./val_label.npy',type=str)
    parse.add_argument('--batch_size',help='batch_size',default=16,type=int)
    parse.add_argument('--ctx',help='mx.gpu(ctx)',default=1,type=int)
    parse.add_argument('--lr',help='learning rate',default=0.0001,type=float)
    parse.add_argument('--lr_type',help='type of learning rate',default='adam',type=str)
    parse.add_argument('--num_epoch',help='number of epoch',default=300,type=int)
    parse.add_argument('--prefix',help='prefix',default='segnet',type=str)
    parse.add_argument('--epoch',help='epoch',default='0',type=int)
    return parse.parse_args()

def dataiter(train_path, train_label_path, val_path, val_label_path, batch_size):

    train = np.load(train_path)
    train_label = np.load(train_label_path)
    
    test = np.load(val_path)
    test_label = np.load(val_label_path)
    
    train_iter = mx.io.NDArrayIter(train,train_label,batch_size=batch_size)
    test_iter = mx.io.NDArrayIter(test,test_label,batch_size=batch_size)
    return train_iter,test_iter

def fit(sym, arg_params, aux_params, train_iter, test_iter, ctx, lr_type, lr, num_epoch, prefix, epoch):
    devs = [mx.gpu(i) for i in range(ctx)]
    mod = mx.mod.Module(sym,context=devs)
    
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("log.txt")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    
    mod.fit(train_data=train_iter,
            eval_data=test_iter,
            batch_end_callback=mx.callback.Speedometer(1,50),
            optimizer=lr_type,
            optimizer_params={'learning_rate':lr},
            initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=True,
            num_epoch = num_epoch,
            kvstore='device'
            )
    mod.save_checkpoint(prefix,epoch)

parse = parse_()    
    
def train(flag=parse.flag,output=parse.output,continue_prefix=parse.continue_prefix,continue_epoch=parse.continue_epoch,train_path=parse.train_path,train_label_path=parse.train_label_path,val_path=parse.val_path,val_label_path=parse.val_label_path,batch_size=parse.batch_size,ctx=parse.ctx,lr=parse.lr,lr_type=parse.lr_type,num_epoch=parse.num_epoch,prefix=parse.prefix,epoch=parse.epoch): 
     
    sym, arg, aux = sym_arg_aux(flag, output, continue_prefix, continue_epoch)
    
    train_iter,test_iter = dataiter(train_path, train_label_path, val_path, val_label_path, batch_size)
    
    fit(sym, arg, aux, train_iter, test_iter, ctx, lr_type, lr, num_epoch, prefix, epoch)
    
   

if __name__ == "__main__":
    train()
	
    
