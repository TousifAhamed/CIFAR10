import torch.nn as nn
import torch.nn.functional as F
import models.base_nn_network_utils as nutils

class Cifar10Net(nn.Module):
    def __init__(self,drop_val=0):
        super(Cifar10Net,self).__init__()

        #Conv block 1
        self.conv_blk1 = nn.Sequential(
            nutils.Conv2d_BasicBlock(inC=3,outC=16,ksize=(3,3),padding=1,drop_val=drop_val), #RF=3x3, outputsize = 32x32x16
            nutils.Conv2d_BasicBlock(inC=16,outC=32,ksize=(3,3),padding=1,drop_val=drop_val), #RF=5x5, outputsize = 32x32x32
            nutils.Conv2d_BasicBlock(inC=32,outC=64,ksize=(3,3),padding=1,drop_val=drop_val), #RF=7x7, outputsize = 32x32x64
        )
        # Transition Layer
        self.transition_blk1 = nutils.Conv2d_TransitionBlock_Dilation(64,32) #RF=11x11, outputsize = 16x16x32

        #Conv block 2
        self.conv_blk2 = nn.Sequential(
            nutils.Conv2d_BasicBlock(inC=32,outC=64,ksize=(3,3),padding=1,dilation=2,drop_val=drop_val), #RF=19x19, outputsize = 14x14x64
            nutils.Conv2d_BasicBlock(inC=64,outC=128,ksize=(3,3),padding=1,dilation=2,drop_val=drop_val), #RF=27x27, outputsize = 12x12x128
        )
        # Transition Layer
        self.transition_blk2 = nutils.Conv2d_TransitionBlock_Dilation(128,16) #RF=35x35, outputsize = 6x6x16

        #Conv block 3
        self.conv_blk3 = nn.Sequential(
            nutils.Conv2d_BasicBlock(inC=16,outC=32,ksize=(3,3),padding=1,drop_val=drop_val), #RF=43x43, outputsize = 6x6x32
            nutils.Conv2d_BasicBlock(inC=32,outC=64,ksize=(3,3),padding=1,drop_val=drop_val), #RF=51x51, outputsize = 6x6x64
        )
        # Transition Layer
        self.transition_blk3 = nutils.Conv2d_TransitionBlock_Dilation(64,16) #RF=67x67, outputsize = 3x3x16

        #Conv block 4
        self.conv_blk4 = nn.Sequential(
            nutils.Conv2d_DepthWiseSeperable_BasicBlock(inC=16,outC=32,ksize=(3,3),padding=1,drop_val=drop_val), #RF=83x83, outputsize = 3x3x32
            nutils.Conv2d_DepthWiseSeperable_BasicBlock(inC=32,outC=64,ksize=(3,3),padding=1,drop_val=drop_val), #RF=99x99, outputsize = 3x3x64
        )

        #output block
        self.output_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=3), # RF=115x115, output size = 1x1x64 
            nn.Conv2d(in_channels=64,out_channels=10,kernel_size=(1,1),padding=0,bias=False) # RF=115x115, output size = 1x1x10
        )

    def forward(self,x):
        x = self.conv_blk1(x)
        x = self.transition_blk1(x)

        x = self.conv_blk2(x)
        x = self.transition_blk2(x)

        x = self.conv_blk3(x)
        x = self.transition_blk3(x)

        x = self.conv_blk4(x)
        x = self.output_block(x)

        x = x.view(-1,10)
        x = F.log_softmax(x)
        return x
        
