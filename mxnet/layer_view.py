#-*-coding:utf-8-*-
import mxnet as mx
import cv2
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
Batch = namedtuple('Batch',['data'])


def get_image(filename,shape=(28,28)):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,shape)
    img = np.swapaxes(img,0,2)
    img = np.swapaxes(img,1,2)
    img = img[np.newaxis,:]
    return img

def model_initial(prefix='model_minist',epoch=20,data_shape=(1,3,28,28),ctx=mx.gpu(0)):
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix,epoch)
    mod = mx.mod.Module(symbol=sym,context=ctx,label_names=None) 
    mod.bind(for_training=False,data_shapes=[('data',data_shape)]) 
    mod.set_params(arg_params,aux_params,allow_missing=True)
    return mod


def output_list(mod):
    lists = mod.symbol.get_internals().list_outputs()
    new_lists = []
    for name in lists:##删除一些没必要可视化的结果  
        if 'bias' in name:
            continue
        elif 'softmax' in name:
            continue
        elif 'fullconnected' in name:
            continue
        elif 'flaten' in name:
            continue
        else:
            new_lists.append(name)
    return new_lists

def output_view(mod,lists,ctx,data_shape,image_path):
    output = [mod.symbol]
    for name in lists:
        output.append(mod.symbol.get_internals()[name])
    group = mx.symbol.Group(output)
    new_model = mx.mod.Module(symbol=group,context=ctx,label_names=None)
    new_model.bind(for_training=False,data_shapes=[('data',data_shape)])
    new_model.set_params(mod.get_params()[0],mod.get_params()[1],allow_missing=True)
    img = get_image(image_path,data_shape[-2:])
    new_model.forward(Batch([mx.nd.array(img)]))
    return new_model.get_outputs()[1:]
    
def view(output,output_name,weight_max):
    layers_view = []
    for i,layer_output in enumerate(output):
        if len(layer_output.shape) != 4:
            print 'this %s layer can not be viewed'%(output_name[i])
            layers_view.append([[0]])
            continue       ######## 没有四个通道的层结构暂时不考虑可视化
        else:
            layer_output = layer_output.asnumpy()
            b,c,x,y = layer_output.shape
            stride = 1
            w_c = int(weight_max/(y+stride))
            if w_c == 0:
                print 'weight_max is lower than the weight of the viewed layer'
                layers_view.append([[0]])
                continue
            h_c = (b*c+w_c-1)/w_c
            
            layer_w = w_c*(y+stride) - stride
            layer_h = h_c*(x+stride) - stride
            
            layer_view = np.zeros(shape=(layer_h,layer_w),dtype=float)
            
            count_w,count_h=0,0
            
            for i in range(b):
                for j in range(c):
                    layer_view[count_h*(x+stride):count_h*(x+stride)+x,count_w*(y+stride):count_w*(y+stride)+y] = layer_output[i][j]
                    count_w +=1
                    if count_w == w_c:
                        count_w = 0
                        count_h += 1
            layers_view.append(layer_view)
    return layers_view
            
def layer_view(mod,lists,ctx=mx.gpu(0),data_shape=(1,3,28,28),image_path='69.png',weight_max = 100):
    result=[]
    output=[]
    idex = []
    for list in lists:
        if list in mod.get_params()[0].keys():
            result.append(mod.get_params()[0][list])
            idex.append(list)
        else:
            output.append(list)
    if len(output) != 0:
        result_ = output_view(mod,output,ctx=ctx,data_shape=data_shape,image_path=image_path)
        for i,out in enumerate(result_):
            result.append(out)
            idex.append(output[i])
    result_view = view(result,idex,weight_max=weight_max)
    ######test 
    for i in range(len(idex)):
        print idex[i],result[i].shape
    ###########
    return idex,result_view
        
def test():
    mod = model_initial('segnet',0001,(1,3,224,224))
    lists = output_list(mod)
    idex,output_view = layer_view(mod,lists,data_shape=(1,3,224,224),image_path='shetou5.jpg',weight_max=3000)
    plt.imshow(output_view[idex.index('conv1_2_bn_D_moving_mean')])
    plt.show()

        
def main():
    mod = model_initial()
    lists = output_list(mod)
    idex,output = layer_view(mod,lists)#lists 是数组输入

if __name__ == '__main__':
    main()