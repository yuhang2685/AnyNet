from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

# YH: A layer sequence "bn + ReLU + Conv2d".
#     I guess the name prefix "pre" suggested the convolution used before some task.
def preconv2d(in_planes, out_planes, kernel_size, stride, pad, dilation=1, bn=True):
    if bn:
        return nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))
    else:
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))

# YH: The model "unetUp"
class unetUp(nn.Module):
    
    def __init__(self, in_size, out_size, is_deconv):
    
        super(unetUp, self).__init__()
        
        self.in_size = in_size
        self.out_size = out_size
        
        # YH: Define "self.up" for "unetUp"
        # YH: What is de-convolution? what is "nn.ConvTranspose2d"?
        if is_deconv:
            self.up = nn.Sequential(
                nn.BatchNorm2d(in_size),
                nn.ReLU(inplace=True),
                # YH: Applies a 2D transposed convolution operator over an input image composed of several input planes.
                #     This module can be seen as the gradient of Conv2d with respect to its input. 
                #     It is also known as a fractionally-strided convolution 
                #     or a deconvolution (although it is not an actual deconvolution operation).
                nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
            )
        else:
            # YH: Applies a "2D bilinear upsampling" to an input signal composed of several input channels.
            #     What is "2D bilinear upsampling"?????
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            in_size = int(in_size * 1.5)

        # YH: Define "self.conv"
        self.conv = nn.Sequential(
            preconv2d(in_size, out_size, 3, 1, 1),
            preconv2d(out_size, out_size, 3, 1, 1),
        )

    def forward(self, inputs1, inputs2):
        # YH: Apply sequential layers "self.up" which is "2D bilinear upsampling" on "inputs2"
        outputs2 = self.up(inputs2)
        # YH: Calculate "buttom and right" as half of inputs1 (image) height and weight.
        buttom, right = inputs1.size(2)%2, inputs1.size(3)%2
        # YH: to pad the last 2 dimensions of the input tensor, then use (padding_left, padding_right, padding_top, padding_bottom) 
        outputs2 = F.pad(outputs2, (0, -right, 0, -buttom))
        # YH: Give the concatenation as input to the 'self.conv' which is a sequence of 'preconv2d'.
        return self.conv(torch.cat([inputs1, outputs2], 1))

# YH: A submodel for AnyNet to extract image featuers.
class feature_extraction_conv(nn.Module):
    
    def __init__(self, init_channels,  nblock=2):
        
        super(feature_extraction_conv, self).__init__()

        self.init_channels = init_channels
        
        nC = self.init_channels
        
        # YH: "downsample_conv" is the first layer, because the input channel is 3 
        #     which is equal to RGB image depth, and "stride = 2" according to the layer summary in the paper.
        #     The arguments for Conv2d is (in_channels, out_channels, kernel_size, stride, padding...)
        downsample_conv = [nn.Conv2d(3,  nC, 3, 1, 1), # 512x256
                                    preconv2d(nC, nC, 3, 2, 1)]
        downsample_conv = nn.Sequential(*downsample_conv)
        
        inC = nC
        outC = 2*nC
        # YH: "_make_block" creates a MaxPool2d(2,2) and #2nblock copies of "preconv2d" layers.
        block0 = self._make_block(inC, outC, nblock)
        # YH: "self.block0" is a down sampling convolutions layer followed by max pooling layer and a sequence of "preconv2d" layers.
        self.block0 = nn.Sequential(downsample_conv, block0)

        nC = 2*nC
        self.blocks = []
        for i in range(2):
            self.blocks.append(self._make_block((2**i)*nC,  (2**(i+1))*nC, nblock))

        self.upblocks = []
        for i in reversed(range(2)):
            # YH: Note here it calls "unetUp" for constructing "self.upblocks"
            self.upblocks.append(unetUp(nC*2**(i+1), nC*2**i, False))

        # YH: "nn.ModuleList" holds sub-modules in a list.
        #     ModuleList can be indexed like a regular Python list, 
        #     but modules it contains are properly registered, and will be visible by all Module methods.
        self.blocks = nn.ModuleList(self.blocks)
        self.upblocks = nn.ModuleList(self.upblocks)

        # YH: Check all self modules and initialize the weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


    # YH: Construct a max pooling layer and a sequence of "preconv2d" layers
    def _make_block(self, inC, outC, nblock ):
        model = []
        model.append(nn.MaxPool2d(2,2))
        for i in range(nblock):
            model.append(preconv2d(inC, outC, 3, 1, 1))
            inC = outC
        return nn.Sequential(*model)


    def forward(self, x):
        
        # YH: Create a sequence of layers to return.
        downs = [self.block0(x)]
        # YH: [block0]
        # YH: [downsample_conv] [self._make_block(inC, outC, nblock)]
        # YH: [downsample_conv] [nn.MaxPool2d(2,2), preconv2d(inC, outC, 3, 1, 1), preconv2d(inC, outC, 3, 1, 1)]
        # YH: [nn.Conv2d(3, nC, 3, 1, 1), preconv2d(nC, nC, 3, 2, 1)] [nn.MaxPool2d(2,2), preconv2d(nC, 2nC, 3, 1, 1), preconv2d(2nC, 2nC, 3, 1, 1)]
        
        # YH: "self.blocks" has 2 parts, each is a [maxpool(2,2) + preconv2d]
        #     It appends self.blocks where each block do what? don't understand if (downs[-1]) is a parameter.
        #     Note "nn.ModuleList" is a "nn.ModuleList"
        for i in range(2): # i = 0, 1
            downs.append(self.blocks[i](downs[-1]))
        downs = list(reversed(downs))
        for i in range(1,3): # i = 1, 2
            downs[i] = self.upblocks[i-1](downs[i], downs[i-1])
        return downs

########################################################
# YH: Below 2 are called for AnyNet after constructing Cost Volume.

# YH: Create a ReLU layer and a Conv3d layer either after layer 'BatchNormalization 3D' or not depends on the parameter "bn3d"
def batch_relu_conv3d(in_planes, out_planes, kernel_size=3, stride=1, pad=1, bn3d=True):
    if bn3d:
        return nn.Sequential(
            # YH: Apply Batch Normalization over the input, and return same shape output.
            nn.BatchNorm3d(in_planes),
            # YH: Apply the ReLU function element-wise.
            nn.ReLU(),
            # YH: 3D convolutions are the generalization of the 2D convolution. 
            #     Here in 3D convolution, the filter depth is smaller than the input layer depth (kernel size < channel size).
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False))
    else:
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False))

# YH: Create a sequence of "batch_relu_conv3d" layers.
#     I guess the name prefix "post" suggested the convolution used after some task.
def post_3dconvs(layers, channels):
    net  = [batch_relu_conv3d(1, channels)]
    net += [batch_relu_conv3d(channels, channels) for _ in range(layers)]
    net += [batch_relu_conv3d(channels, 1)]
    return nn.Sequential(*net)

