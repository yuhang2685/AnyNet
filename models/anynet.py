from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodules import post_3dconvs,feature_extraction_conv
import sys

'''
    YH: Computational graphs and autograd are a very powerful paradigm 
        for defining complex operators and automatically taking derivatives; 
        however for large neural networks raw autograd can be a bit too low-level.
        When building neural networks we frequently think of arranging the computation into layers, 
        some of which have learnable parameters which will be optimized during learning.
        
        In TensorFlow, packages like Keras, TensorFlow-Slim, and TFLearn 
        provide higher-level abstractions over raw computational graphs that are useful for building neural networks.
        In PyTorch, the nn package serves this same purpose. 
        The nn package defines a set of Modules, which are roughly equivalent to neural network layers.
        A Module receives input Tensors and computes output Tensors, 
        but may also hold internal state such as Tensors containing learnable parameters. 
        The nn package also defines a set of useful loss functions that are commonly used when training neural networks.
'''
# AnyNet is of type nn.Module which is the base class for all neural network modules.
class AnyNet(nn.Module):
    ''' 
    YH: "__init__" is a reseved method in python classes. 
        It is known as a constructor in object oriented concepts. 
        This method called when an object is created from the class 
        and it allow the class to initialize the attributes of a class.
    '''
    def __init__(self, args):
    
        # YH: 'super' of AnyNet is 'nn.Module'
        super(AnyNet, self).__init__()
        
        # YH: Initialize AnyNet attributes with given parameters 'args'.
        self.init_channels = args.init_channels
        self.maxdisplist = args.maxdisplist
        self.spn_init_channels = args.spn_init_channels
        self.nblocks = args.nblocks
        self.layers_3d = args.layers_3d
        self.channels_3d = args.channels_3d
        self.growth_rate = args.growth_rate
        self.with_spn = args.with_spn

        # YH: If it includes a 'spn' module for stage_4.
        if self.with_spn:
            
            try:
                # from .spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind
                from .spn_t1.modules.gaterecurrent2dnoind import GateRecurrent2dnoind
            except:
                print('Cannot load spn model')
                sys.exit()
            
            # YH: Define sub-model attribute 'spn_layer'.
            self.spn_layer = GateRecurrent2dnoind(True,False)

            spnC = self.spn_init_channels
            
            '''
            YH: Conv1D is used for input signals which are similar to the voice. 
                Conv2D is used for images. 
                Conv3D is usually used for videos where you have a frame for each time span.
            '''
            
            # YH: Define sub-model attribute 'refine_spn'.
            self.refine_spn = [nn.Sequential(
                nn.Conv2d(3, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*3, 3, 1, 1, bias=False),
            )]
            self.refine_spn += [nn.Conv2d(1,spnC,3,1,1,bias=False)]
            self.refine_spn += [nn.Conv2d(spnC,1,3,1,1,bias=False)]
            self.refine_spn = nn.ModuleList(self.refine_spn)
        else:
            self.refine_spn = None

        # YH: Initialize 'feature_extraction' as a sub-model attribute of AnyNet.
        self.feature_extraction = feature_extraction_conv(self.init_channels,
                                      self.nblocks)

        # YH: Define attribute "volume_postprocess" as a sub-model of 3 "post_3dconvs" layers.
        self.volume_postprocess = []

        for i in range(3):
            net3d = post_3dconvs(self.layers_3d, self.channels_3d*self.growth_rate[i])
            self.volume_postprocess.append(net3d)
        self.volume_postprocess = nn.ModuleList(self.volume_postprocess)


        '''
        YH: Modules can also contain other Modules, allowing to nest them in a tree structure. 
            You can assign the submodules as regular attributes (properties, fields).
        
            The self.modules() method returns an iterable 
            to the many layers or “modules” defined in the model class. 
            This particular piece of code is using that self.modules() iterable 
            to initialize the weights of the different layers present in the model.
            isinstance() checks if the particular layer “m” is an instance 
            of a conv2d or linear or conv3d layer etc. and initializes the weights accordingly.
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    # YH: 'warp' is "up_sampling" the disparity of previous stage (stored in matrix 'disp') to image 'x'.
    # YH: It is called in '_build_volume_2d3'.
    def warp(self, x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W, device='cuda').view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device='cuda').view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        vgrid = torch.cat((xx, yy), 1).float()

        # vgrid = Variable(grid)
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        return output

    # YH: This function is called in 'forward'.
    # YH: It constructs a Cost Volume from the extracted features 'feat_l' and 'feat_right'.
    def _build_volume_2d(self, feat_l, feat_r, maxdisp, stride=1):
        assert maxdisp % stride == 0  # Assume maxdisp is multiple of stride
        # YH: Initialize a matrix for Cost Volume.
        cost = torch.zeros((feat_l.size()[0], maxdisp//stride, feat_l.size()[2], feat_l.size()[3]), device='cuda')
        # YH: The last dimension of Cost Volume is for different "disparity" less than 'max_disparity'.
        for i in range(0, maxdisp, stride):
            cost[:, i//stride, :, :i] = feat_l[:, :, :, :i].abs().sum(1)
            if i > 0:
                cost[:, i//stride, :, i:] = torch.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], 1, 1)
            else:
                cost[:, i//stride, :, i:] = torch.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], 1, 1)

        return cost.contiguous()

    # YH: The function is called in forward for constructing Cost Volume using extracted featuers and the warped disparity of the previous stage .
    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):
        size = feat_l.size()
        batch_disp = disp[:,None,:,:,:].repeat(1, maxdisp*2-1, 1, 1, 1).view(-1,1,size[-2], size[-1])
        batch_shift = torch.arange(-maxdisp+1, maxdisp, device='cuda').repeat(size[0])[:,None,None,None] * stride
        batch_disp = batch_disp - batch_shift.float()
        batch_feat_l = feat_l[:,None,:,:,:].repeat(1,maxdisp*2-1, 1, 1, 1).view(-1,size[-3],size[-2], size[-1])
        batch_feat_r = feat_r[:,None,:,:,:].repeat(1,maxdisp*2-1, 1, 1, 1).view(-1,size[-3],size[-2], size[-1])
        # YH: Here is the place that warped the already upsampled previous disparity is integrated into the "Cost Volume"
        cost = torch.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), 1, 1)
        cost = cost.view(size[0],-1, size[2],size[3])
        return cost.contiguous()


    '''
    YH: "Forward" defines the computation performed at every call.  
        Should be overridden by all subclasses.
    '''
    def forward(self, left, right):

        img_size = left.size()

        # YH: The first step of AnyNet is to extract features of Left_image and Right_image.
        feats_l = self.feature_extraction(left)
        feats_r = self.feature_extraction(right)
        
        pred = []
        
        # YH: Then we compute disparity for different scales in the range of [0, len(feats_l)].
        for scale in range(len(feats_l)):
        
            # YH: Create Cost Volumes for different scale.
            if scale > 0:
                # YH: Upsample the result disparity of previous stage by bilinear
                wflow = F.upsample(pred[scale-1], (feats_l[scale].size(2), feats_l[scale].size(3)),
                                   mode='bilinear') * feats_l[scale].size(2) / img_size[2]
                cost = self._build_volume_2d3(feats_l[scale], feats_r[scale],
                                         self.maxdisplist[scale], wflow, stride=1)
            else:
                # YH: 'scale=0' is Stage_1 because there is no disparity from the previous stage to be warped.
                cost = self._build_volume_2d(feats_l[scale], feats_r[scale],
                                             self.maxdisplist[scale], stride=1)

            cost = torch.unsqueeze(cost, 1)
            cost = self.volume_postprocess[scale](cost)
            cost = cost.squeeze(1)
            
            # YH: Regression to transform 'Cost Volume' to 'disparity'.
            # YH: If it is stage_1, do not need to work with the previous stage output disparity. 
            if scale == 0:
                pred_low_res = disparityregression2(0, self.maxdisplist[0])(F.softmax(-cost, dim=1))
                pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
                disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')
                pred.append(disp_up)
            # YH: If it is not stage_1, need to work with the previous stage output disparity.
            else:
                pred_low_res = disparityregression2(-self.maxdisplist[scale]+1, self.maxdisplist[scale], stride=1)(F.softmax(-cost, dim=1))
                pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
                disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')
                pred.append(disp_up+pred[scale-1])

        # YH: If AnyNet with SPN, then refine_spn is not None.
        if self.refine_spn:
            spn_out = self.refine_spn[0](nn.functional.upsample(left, (img_size[2]//4, img_size[3]//4), mode='bilinear'))
            G1, G2, G3 = spn_out[:,:self.spn_init_channels,:,:], spn_out[:,self.spn_init_channels:self.spn_init_channels*2,:,:], spn_out[:,self.spn_init_channels*2:,:,:]
            sum_abs = G1.abs() + G2.abs() + G3.abs()
            G1 = torch.div(G1, sum_abs + 1e-8)
            G2 = torch.div(G2, sum_abs + 1e-8)
            G3 = torch.div(G3, sum_abs + 1e-8)
            # YH: Here is the place to get the disparity from the previous stage.
            pred_flow = nn.functional.upsample(pred[-1], (img_size[2]//4, img_size[3]//4), mode='bilinear')
            refine_flow = self.spn_layer(self.refine_spn[1](pred_flow), G1, G2, G3)
            refine_flow = self.refine_spn[2](refine_flow)
            pred.append(nn.functional.upsample(refine_flow, (img_size[2] , img_size[3]), mode='bilinear'))


        return pred


# YH: I guess the name suggesting it is the "regression" for transforming "Cost Volume" to "Disparity Map".
class disparityregression2(nn.Module):
    
    def __init__(self, start, end, stride=1):
        
        super(disparityregression2, self).__init__()
        '''
        YH: Numpy is a great framework, but it cannot utilize GPUs to accelerate its numerical computations.
            A PyTorch Tensor is conceptually identical to a numpy array: a Tensor is an n-dimensional array, 
            and PyTorch provides many functions for operating on these Tensors. 
            Behind the scenes, Tensors can keep track of a computational graph and gradients, 
            but they’re also useful as a generic tool for scientific computing.
            Also unlike numpy, PyTorch Tensors can utilize GPUs to accelerate their numeric computations. 
            To run a PyTorch Tensor on GPU, you simply need to specify the correct device.
            When using autograd, the forward pass of your network will define a computational graph; 
            nodes in the graph will be Tensors, and edges will be functions that produce output Tensors from input Tensors. 
            Backpropagating through this graph then allows you to easily compute gradients.
            Each Tensor represents a node in a computational graph. 
            If x is a Tensor that has x.requires_grad=True 
            then x.grad is another Tensor holding the gradient of x with respect to some scalar value.
            
            A torch.Tensor is a multi-dimensional array containing elements of a single data type.
            As I understood, more than that, it can be specified to using GPU (by 'CUDA') or CPU.
            
            A Tensor can be created from python Data types and converted back with ease.
            
            The following code initiates 'self.disp' as a Tensor by 'torch.arange' 
            which returns a 1-D tensor of size '(end*stride-start*stride)/stride'
            with values from the interval [start*stride, end*stride]
            taken with common difference step beginning from start.
        '''
        self.disp = torch.arange(start*stride, end*stride, stride, device='cuda', requires_grad=False).view(1, -1, 1, 1)


        '''
        YH: Under the hood, each primitive autograd operator is really two functions that operate on Tensors. 
            The forward function computes output Tensors from input Tensors. 
            The backward function receives the gradient of the output Tensors with respect to some scalar value, 
            and computes the gradient of the input Tensors with respect to that same scalar value.
            
            In PyTorch we can easily define our own autograd operator 
            by defining a subclass of 'torch.autograd.Function'
            and implementing the forward and backward functions. 
            We can then use our new autograd operator by constructing an instance 
            and calling it like a function, passing Tensors containing input data.
        '''
    def forward(self, x):
        # YH: It repeats 'self.disp' items (duplicate and concatenated?) along the specified dimensions.
        #     Unlike expand(), this function copies the tensor’s data.
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        # YH: Sum the matrix by rows.
        out = torch.sum(x * disp, 1, keepdim=True)
        return out