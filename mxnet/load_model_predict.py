#-*- coding:UTF-8 -*-
import mxnet as mx 
import cv2
import numpy as np
from collections import namedtuple
import os

batch_size = 160

class mxnet_test(object):
    def __init__(self, model_name, epoch):
        self.model_name = model_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.image = None
        self.dataiter = None
        self.mod = None
        self.size = 224
        self.test_batch = 4
        
        
    def load_model_1(self):
        #加载模型显示各层名字
        model = mx.module.Module.load(self.model_name, self.epoch)
        internals = model.symbol.get_internals()
        print internals['conv1_1_weight'].data
        #print internals.list_outputs()
        
    def load_model(self):
        sym, arg_params,aux_params = mx.model.load_checkpoint(self.model_name,self.epoch)
        mod= mx.mod.Module(symbol=sym, context=mx.gpu())
        mod.bind(for_training=False, data_shapes=[('data', (1,3,self.size,self.size))])
        mod.set_params(arg_params, aux_params)
        self.mod = mod
        
    def predict(self):
        #load_data 模式  NDarrayIter 转 numpy
        self.mod.predict(self.dataiter)
        print self.mod.get_outputs()
        prob = self.mod.get_outputs()[0].asnumpy()
        print prob.shape
        
    def predict_v2(self):
        #load_data_v2 模式 Batch 转 NDArray
        self.mod.forward(self.dataiter,is_train = False)
        prob = self.mod.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        #a = np.argsort(prob)[::-1]
        return prob
        
    def predict_list(self):
        out = self.mod.predict(self.dataiter)
        print out
    def load_data(self,path):
        self.get_image(path)
        self.dataiter = mx.io.NDArrayIter(self.image)
        
    def load_data_v2(self,path):
        self.get_image(path)
        Batch= namedtuple('Batch', ['data'])
        self.dataiter = Batch([mx.nd.array(self.image)])
        
    def load_data_list(self,path):
        if os.path.isdir(path):
            dir = os.listdir(path)
            images = np.ndarray(shape=(len(dir),3,self.size,self.size), dtype=np.float32)
            i = 0 
            for fi in dir: 
                self.get_image(os.path.join(path,fi))
                images[i,:,:,:] = self.image
                i += 1
            self.dataiter = mx.io.NDArrayIter(images, batch_size = self.test_batch )
            
    def get_image(self, path):
        img = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
        #->(224,224,3) -> (3,224,224)
        img = cv2.resize(img, (self.size, self.size))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]
        self.image = img
        #print img.shape
        
    def get_image_gray(self,path):
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(self.size,self.size),0)
        img = img[np.newaxis, :]
        img = img[np.newaxis, :]
        self.image = img
        
    def visual(self):
        sym, arg_params,aux_params = mx.model.load_checkpoint(self.model_name,self.epoch)
        data_shape = (1,3,224,224)
        mx.viz.plot_network(sym, shape={"data":data_shape}, node_attrs={"shape":'oval',"fixedsize":'false'}).view()
    
    def save(self):
        model=mx.model.FeedForward.load('vgg-face',1,num_batch_size=1)
        internals = model.symbol.get_internals()
        #print internals.list_outputs()
        fea_symbol = internals['fc7_output']
        feature_extractor=mx.model.FeedForward(symbol=fea_symbol,numpy_batch_size=1,arg_params=model.arg_params,aux_params=model.aux_params,allow_extra_params=True) 
        feature_extractor.save('new_model',1)
        
    def arg_aux(self):
        sym, arg_params,aux_params = mx.model.load_checkpoint(self.model_name,self.epoch)
        arg_name = sym.list_arguments()
        print len(arg_name)
        print arg_name
        #for i,x in arg_params.items():
        #    print i  
        #   print x.shape
        #print len(arg_params)
        #print arg_params.keys()
        #print len(aux_params)
        #print aux_params.keys()
        #data_shape=(1,3,500,500)
        #arg_shapes, _, _ = sym.infer_shape(data=data_shape)
	del arg_params['fc6_weight']
	del arg_params['fc6_bias']
	del arg_params['fc7_bias']
	del arg_params['fc7_weight']
	del arg_params['fc8_weight']
	del arg_params['fc8_bias']
	for name in arg_name:
		if arg_params.has_key(name):
			print name
			print arg_params[name].shape
		else :
			print name
	return arg_params
     
    def arg_aux_1(self):
        sym, arg_params,aux_params = mx.model.load_checkpoint(self.model_name,self.epoch)
        arg_name = sym.list_arguments()
        del arg_params['fc6_weight']
        del arg_params['fc6_bias']
        del arg_params['fc7_bias']
        del arg_params['fc7_weight']
        del arg_params['fc8_weight']
        del arg_params['fc8_bias']
        
        arg = {}
        for name in arg_name:
            if arg_params.has_key(name):
                #if name[-4:] == 'bias':
                 #   new_name = name[:7] + '_D' + name[7:]
                 #   h = dict([(new_name,arg_params[name])])
                 #   arg.update(h)
                    #print new_name
                if name[-4:] == 'ight':
                    new_name = name[:7] + '_D' + name[7:]
                    h = dict([(new_name,np.swapaxes(arg_params[name],0,1))])
                    arg.update(h)
                    #print new_name
            
        #print arg.keys()
        #for name in arg_name:
        #    if arg_params.has_key(name): 
        #        new_name = name[:7] + '_D' + name[7:]
         #       print new_name
         #       print arg[new_name].shape
        del arg['conv1_1_D_weight']
        
        
        return arg
        
      
class vgg_c(mxnet_test):
    def __init__(self,model_name,epoch):
        super(vgg_c,self).__init__(model_name,epoch)
        self.path = parse.path
    def predict_vgg(self):
        self.load_model()
        fi  = file('./val_label.txt','r')
        result = file('./result/%s-%04d.txt'%(self.model_name.split('/')[-1],self.epoch),'w')
        for line in fi.readlines():
            dir = line.split(' ')[0]
            label = line.split(' ')[1].split('\n')[0]
            path_dir = os.path.join(self.path,dir)
            if os.path.isdir(path_dir):
                lists = os.listdir(path_dir)
                #x = np.ndarray((1,len(lists)),dtype=float)
                x = np.zeros((len(lists),7),dtype=float)
                i = 0;
                for list in lists:
                    self.load_data_v2(os.path.join(path_dir,list))
                    x[i,:] = self.predict_v2()
                    i+= 0
                print np.mean(x,axis=0)
                re = np.argsort(np.mean(x,axis=0))[::-1][0]
                print re
                result.write(dir+' '+str(re)+' '+label+'\n')
        fi.close()
        result.close()
                    
        
 
#vgg_lstm = mxnet_test('new', 2)
#vgg_lstm.load_data('1.jpg')mxnet
#vgg_lstm.load_data_v2('./data/Val_2018_preprocessing/000123880/000001.jpg')
#vgg_lstm.load_model()
#vgg_lstm.predict_v2()
def main():
    model = './model/afew_36'
    epoch = 3
    #path = './data/val_mtcnn_ori'
    path = './data/Val_2018_preprocessing'
    import argparse

    parse = argparse.ArgumentParser(description='predict model!~')
    parse.add_argument('--model',help='model name',default=model,type=str)
    parse.add_argument('--epoch',help='model epoch',default=epoch,type=int)
    parse.add_argument('--path',help='model epoch',default=path,type=str)
    parse = parse.parse_args()


    vgg = vgg_c(parse.model,parse.epoch)
    vgg.predict_vgg()

        
