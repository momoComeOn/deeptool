import mxnet as mx
import sys
#train_iter = './data/fer2013_train.rec'
#train_iter = './data/train_raf.rec'
#train_iter = './data/train_raf_single.rec'
#train_iter = "./data/train_lbp.rec"
#train_iter = './data/train.rec'
#train_iter = './data/train_val.rec'
#train_iter = './data/train_mtcnn.rec'
#train_iter = './data/train_mtcnn_filter.rec'
train_iter = './data/train_face_background.rec'
#train_iter = './data/train_mtcnn_extent.rec'
#val_iter = "./data/val_lbp.rec"
#val_iter = './data/test_train.rec'
#val_iter = './data/val.rec'
#val_iter = './data/val_raf_single.rec'
#val_iter = "./data/val_raf.rec"
#val_iter = "./data/val_mtcnn.rec"
#val_iter = "./data/val_mtcnn_filter.rec"
val_iter = './data/val_face_background.rec'
#val_iter = "./data/fer2013_val.rec"
import argparse


parse = argparse.ArgumentParser(description='train model!')
parse.add_argument('--prefix',help='prefix',default='./model/Inception-7',type=str)
parse.add_argument('--epoch',help='epoch',default=1,type=int)
parse.add_argument('--save',help='save',default='./model/v3_face',type=str)
parse.add_argument('--num_epoch',help='number epoch',default=10,type=int)
parse.add_argument('--lr',help='number epoch',default=0.01,type=float)
parse = parse.parse_args()

import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("./log/%s-%04d.txt"%(parse.save.split('/')[-1],parse.num_epoch))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
batch_size=32
train_dataiter = mx.io.ImageRecordIter(
            path_imgrec=train_iter,
            resize=299,
            #mean_img=datadir+"/mean.bin",
            rand_crop=False,
            rand_mirror=True,
#            mean_r = 104,
#	    mean_g = 117,
#	    mean_b = 123,
	    mean_r = 128,
	    mean_g = 128,
	    mean_b = 128,
	    scale = 1/128.0,
	    shuffle= True,
            data_shape=(3,299,299),
            batch_size=batch_size,
            preprocess_threads=4)
test_dataiter = mx.io.ImageRecordIter(
            path_imgrec=val_iter,
            resize=299,
            #mean_img=datadir+"/mean.bin",
            rand_crop=False,
            rand_mirror=False,
            shuffle= False,
 #           mean_r = 104,
#	    mean_g = 117,
#	    mean_b = 123,
	    mean_r = 128,
	    mean_g = 128,
	    mean_b = 128,
	    scale = 1/128.0,
            data_shape=(3,299,299),
            batch_size=batch_size,
            preprocess_threads=4)

sym,arg_params,aux_params=mx.model.load_checkpoint(parse.prefix,parse.epoch)


def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    #devs = [mx.gpu(i) for i in range(num_gpus)]
    devs = mx.gpu(0)
    metric = [mx.metric.Accuracy(),mx.metric.CrossEntropy()]
    mod = mx.mod.Module(symbol=symbol, context=devs)
    mod.fit(train, val,
        num_epoch=parse.num_epoch,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 200),
        #epoch_end_callback = mx.callback.do_checkpoint(parse.save,1),
        kvstore='device',
        #optimizer='sgd',
        #optimizer_params={'learning_rate':parse.lr,"momentum":0.9},
       	optimizer = 'adam',
	optimizer_params = {'learning_rate':parse.lr,"beta1":0.9,"beta2":0.999,"epsilon":1e-08},
	initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        #eval_metric='acc')
	eval_metric = metric)
    mod.save_checkpoint(parse.save,parse.num_epoch)
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)
mod_score = fit(sym, arg_params, aux_params, train_dataiter, test_dataiter, batch_size, 2)
