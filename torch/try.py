import torch
import torch.nn as nn
from torchvision import models
import Dataset
import time
from torch.autograd import Variable

class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()

        base_model = models.vgg19(pretrained=True)
        base_models = list(base_model.features)

        # print (base_models)
        base_classifier = list(base_model.classifier.children())
        # print (base_classifier)
        self.classifier = nn.Sequential(*base_models,View(-1,512*7*7),*base_classifier[:-1],nn.Linear(4096,num_classes))



    def forward(self,x):
        return nn.functional.log_softmax(self.classifier(x),dim=1)


def main():
    vgg = VGG(7)
    vgg = vgg.cuda()

    optimizer = torch.optim.Adam(vgg.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    train_loader = Dataset.data_get(8)
    # import pdb;pdb.set_trace()
    for epoch in range(10):
        start = time.time()
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x, requires_grad=False)
            b_y = Variable(y, requires_grad=False)
            b_x = b_x.cuda()
            b_y = b_y.cuda().long()

            output = vgg(b_x)
            loss = loss_function(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('Epoch:', epoch, '|Step:', step,
                      '|train loss:%.4f'%loss.data[0])
        duration = time.time() - start
        print('Training duation: %.4f'%duration)

main()
