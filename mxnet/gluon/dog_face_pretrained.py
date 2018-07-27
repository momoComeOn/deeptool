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

def zip_extract_data():
	zipfiles= ['train.zip', 'test.zip', 'labels.csv.zip']
	data_dir = '../data/kaggle_dog'
	for f in zipfiles:
    		with zipfile.ZipFile(data_dir + '/' + f, 'r') as z:
        		z.extractall(data_dir)

#zip_extract_data()



def reorg_dog_data(data_dir='train', label_file='labels.csv', train_dir='train', test_dir='test', input_dir='train_valid_test',valid_ratio=0.1):
    # 读取训练数据标签。
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # 跳过文件头行（栏名称）。
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict(((idx, label) for idx, label in tokens))
    labels = set(idx_label.values())

    n_train = len(os.listdir(os.path.join(data_dir, train_dir)))
    # 训练集中数量最少一类的狗的样本数。
    min_n_train_per_label = (
        collections.Counter(idx_label.values()).most_common()[:-2:-1][0][1])
    # 验证集中每类狗的样本数。
    n_valid_per_label = math.floor(min_n_train_per_label * valid_ratio)
    label_count = {}

    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    # 整理训练和验证集。
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = train_file.split('.')[0]
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))

    # 整理测试集。
    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))
def data_zhengli():
    reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir,
                   valid_ratio)
#data_zhengli()




transform_train = gdata.vision.transforms.Compose([
    # 随机对图片裁剪出面积为原图片面积 0.08 到 1 倍之间、且高和宽之比在 3/4 和 4/3 之间
    # 的图片，再放缩为高和宽均为 224 像素的新图片。
    gdata.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                              ratio=(3.0/4.0, 4.0/3.0)),
    # 随机左右翻转图片。
    gdata.vision.transforms.RandomFlipLeftRight(),
    # 随机抖动亮度、对比度和饱和度。
    gdata.vision.transforms.RandomColorJitter(brightness=0.4, contrast=0.4,
                                              saturation=0.4),
    # 随机加噪音。
    gdata.vision.transforms.RandomLighting(0.1),

    # 将图片像素值按比例缩小到 0 和 1 之间，并将数据格式从“高 * 宽 * 通道”改为“通道 * 高 * 宽”。
    gdata.vision.transforms.ToTensor(),
    # 对图片的每个通道做标准化。
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])
])

# 测试时，只使用确定性的图像预处理操作。
transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),
    # 将图片中央的高和宽均为 224 的正方形区域裁剪出来。
    gdata.vision.transforms.CenterCrop(224),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])
])

data_dir = '/home/muyouhang/zkk/new_vgg_lstm/data'
#label_file = 'labels.csv'
#train_dir = 'train'
#test_dir = 'test'
batch_size = 128
valid_ratio = 0.1
# 读取原始图像文件。flag=1 说明输入图像有三个通道（彩色）。
train_ds = gdata.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train_noface_background'), flag=1)
valid_ds = gdata.vision.ImageFolderDataset(
    os.path.join(data_dir, 'val_noface_background'), flag=1)
#train_valid_ds = gdata.vision.ImageFolderDataset(
#    os.path.join(data_dir, input_dir, 'train_valid'), flag=1)
#test_ds = gdata.vision.ImageFolderDataset(
#    os.path.join(data_dir, input_dir, 'test'), flag=1)

train_data = gdata.DataLoader(train_ds.transform_first(transform_train),
                              batch_size, shuffle=True, last_batch='keep')
valid_data = gdata.DataLoader(valid_ds.transform_first(transform_test),
                              batch_size, shuffle=True, last_batch='keep')
#train_valid_data = gdata.DataLoader(train_valid_ds.transform_first(
#    transform_train), batch_size, shuffle=True, last_batch='keep')
#test_data = gdata.DataLoader(test_ds.transform_first(transform_test),
#                             batch_size, shuffle=False, last_batch='keep')

def get_net(ctx):
    # 设 pretrained=True 就能获取预训练模型的参数。第一次使用时需要联网下载。
    finetune_net = model_zoo.vision.vgg16(pretrained=True,root='./model')
    # 定义新的输出网络。
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # 120 是输出的类别数。
    finetune_net.output_new.add(nn.Dense(7))
    # 初始化输出网络。
    finetune_net.output_new.initialize(init.Xavier(), ctx=ctx)
    # 把模型参数分配到即将用于计算的 CPU 或 GPU 上。
    finetune_net.collect_params().reset_ctx(ctx)
    return finetune_net


loss = gloss.SoftmaxCrossEntropyLoss()

def get_loss(data, net, ctx):
    l = 0.0
    for X, y in data:
        y = y.as_in_context(ctx)
        # 计算预训练模型输出层的输入，即特征。
        output_features = net.features(X.as_in_context(ctx))
        # 将特征作为我们定义的输出网络的输入，计算输出。
        outputs = net.output_new(output_features)
        l += loss(outputs, y).mean().asscalar()
    return l / len(data)

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period,
          lr_decay):
    # 只训练我们定义的输出网络。
    trainer = gluon.Trainer(net.output_new.collect_params(), 'adam',
                            {'learning_rate': lr})
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_l = 0.0
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for X, y in train_data:
            y = y.astype('float32').as_in_context(ctx)
            # 计算预训练模型输出层的输入，即特征。
            output_features = net.features(X.as_in_context(ctx))
            with autograd.record():
                # 将特征作为我们定义的输出网络的输入，计算输出。
                outputs = net.output_new(output_features)
                l = loss(outputs, y)
            # 反向传播只发生在我们定义的输出网络上。
            l.backward()
            trainer.step(batch_size)
            train_l += l.mean().asscalar()
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_s = "time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = get_loss(valid_data, net, ctx)
            epoch_s = ("epoch %d, train loss %f, valid loss %f, "
                       % (epoch, train_l / len(train_data), valid_loss))
        else:
            epoch_s = ("epoch %d, train loss %f, "
                       % (epoch, train_l / len(train_data)))
        prev_time = cur_time
        print(epoch_s + time_s + ', lr ' + str(trainer.learning_rate))

ctx = gb.try_gpu()
num_epochs = 10
# 学习率。
lr = 0.001
# 权重衰减参数。
wd = 1e-4
# 优化算法的学习率将在每 10 个迭代周期时自乘 0.1。
lr_period = 5
lr_decay = 0.1


net = get_net(ctx)
net.hybridize()
train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period, lr_decay)
