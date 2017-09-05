import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = ['ResNet', 'resnet18', 'rn_builder']


class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        x =  torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)
        return x

class MyModuleList(nn.ModuleList):
    def __add__(self, x):
        tmp = [m for m in self.modules()] + [m for m in x.modules()]
        return MyModuleList(tmp)
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

def make_basic_block(inplanes, planes, stride=1, downsample=None):
    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)
    block_list = MyModuleList([
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
    ])
    
    if downsample == None:
        residual = MyModuleList([])
    else:
        residual = downsample
    return (block_list, residual)

def make_bottleneck_block(inplanes, planes, stride=1, downsample=None):
    block_list = MyModuleList([
            # conv bn relu
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            # conv bn relu
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            # conv bn
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
    ])
    if downsample == None:
        residual = MyModuleList([])
    else:
        residual = downsample
    return (block_list, residual)

class ResNet(nn.Module):
    def __init__(self, section_reps, num_classes=1000, 
                 conv1_size=7, nbf=64,
                 downsample_start=True,
                 use_basic_block=True,
                 train_death_rate=None,
                 test_death_rate=None):
        super(ResNet, self).__init__()

        if train_death_rate == None:
            self.train_death_rate = [[0.0] * x for x in section_reps]
        else:
            self.train_death_rate = train_death_rate
        if test_death_rate == None:

            self.test_death_rate = [[0.0] * x for x in section_reps]
        else:
            self.test_death_rate = test_death_rate
        if not all(map(lambda i: len(self.train_death_rate[i]) == section_reps[i],
                       range(len(section_reps)))):
            raise Exception('Train death rates do not match size')
        if not all(map(lambda i: len(self.test_death_rate[i]) == section_reps[i],
                       range(len(section_reps)))):
            raise Exception('Test death rates do not match size')

        if use_basic_block:
            self.expansion = 1
            self.block_fn = make_basic_block
        else:
            self.expansion = 4
            self.block_fn = make_bottleneck_block
        self.downsample_start = downsample_start
        self.inplanes = nbf

        self.conv1 = nn.Conv2d(3, nbf, kernel_size=conv1_size,
                               stride=downsample_start + 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nbf)
        self.sections = []
        for i, section_rep in enumerate(section_reps):
            self.sections.append(self._make_section(nbf * (2 ** i), section_rep, stride=(i != 0) + 1))
        lin_inp = nbf * int(2 ** (len(section_reps) - 1)) * self.expansion \
            if len(self.sections) != 0 else nbf
        self.fc = nn.Linear(lin_inp, num_classes)

        self.update_modules()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def update_modules(self):
        # PyTorch requires the layers to be registered for propogation purposes.
        # If we ever change the layers, everything goes to shit. So update it
        self.registered = MyModuleList([])
        for section in self.sections:
            for block, shortcut in section:
                self.registered.append(block)
                self.registered.append(shortcut)

    def _make_section(self, planes, num_blocks, stride=1):
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = MyModuleList([DownsampleB(self.inplanes, planes * self.expansion, stride)])         
        else:
            downsample = None

        blocks = []
        blocks.append(self.block_fn(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * self.expansion
        for i in range(1, num_blocks):
            blocks.append(self.block_fn(self.inplanes, planes))

        return blocks

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        if self.downsample_start:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        death_rates = self.train_death_rate if self.training else self.test_death_rate
        # if not self.training:
        #     print ("printing death rates in testing mode: ")
        #     print (death_rates)
        for sec_ind, section in enumerate(self.sections):
            for block_ind, (block, shortcut) in enumerate(section):
                dr = death_rates[sec_ind][block_ind]
                x_input = x
                if len(shortcut) != 0:
                    x = shortcut(x)
                if torch.rand(1)[0] >= dr:
                    x_conv = block(x_input) / (1. - dr)
                    x = x + x_conv
                    x = F.relu(x)
      
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# Only basic block for now
def rn_builder(section_reps, **kwargs):
    return ResNet(section_reps, **kwargs)

def createModel(depth, data, num_classes, death_mode='none', death_rate=0.5, 
                test_death_mode='none', **kwargs):
    n = (depth - 2) // 6 
    section_reps=[n]*3 #corresponding to 3 _make_layer 

    nblocks = (depth - 2) // 2

    def output_death_rates(mode, death_rate, section_reps):
        #output a range of death rates for the model based on initial dr and death mode
        death_rates = []
        if mode == 'uniform':
            death_rates_list = [death_rate] * nblocks 
        elif mode == 'linear':
            death_rates_list = [float(i + 1) * death_rate / float(nblocks)
                           for i in range(nblocks)]
        else:
            death_rates = None

        if mode == "uniform" or mode == "linear":
            count = 0
            for i in range(len(section_reps)): 
                death_rates.append(death_rates_list[count:(count+section_reps[i])]) 
                count += section_reps[i]
        return death_rates

    train_death_rate = output_death_rates(death_mode, death_rate, section_reps)
    #test_death_rates = output_death_rates(test_death_mode, test_death_rate, section_reps)
    if 'test_death_rate' in kwargs:
        test_death_rate = kwargs['test_death_rate']
    else:
        test_death_rate = None
    model = ResNet(section_reps=section_reps, num_classes=num_classes, 
                    conv1_size=3, downsample_start=False, train_death_rate=train_death_rate, 
                    nbf=16, test_death_rate=test_death_rate)#, test_death_rate=test_death_rates)
    return model

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet([2, 2, 2, 2], **kwargs)
    return model

