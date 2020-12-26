import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
import utils.logger as logger

import models.anynet

'''
YH: PyTorch has long been the preferred deep-learning library for researchers, 
    while TensorFlow is much more widely used in production. 
    PyTorch's ease of use combined with the default eager execution mode for easier debugging 
    predestines it to be used for fast, hacky solutions and smaller-scale models.
'''

parser = argparse.ArgumentParser(description='AnyNet with Flyingthings3d')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datapath', default='dataset/', help='datapath')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6, help='batch size for training (default: 12)')
parser.add_argument('--test_bsize', type=int, default=4, help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='train_results', help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None, help='resume path')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels of the 3d network')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers of the 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')

'''
YH: The "argparse" module makes it easy to write user-friendly command-line interfaces.
    The program defines what arguments it requires, and "argparse" will figure out how to parse those out of "sys.argv". 
    The argparse module also automatically generates help and usage messages 
    and issues errors when users give the program invalid arguments.
'''
args = parser.parse_args()

'''
YH: This file is mainly for training. 
    It contains testing which is called once the training is over, 
    which serves for showing the estimation for the accuracy of the trained model on unseen data.
'''
def main():
    
    global args

    # YH: First load files of SecenFlow dataset.
    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
        args.datapath)

    # YH: Create a "DataLoader" instance for training.
    TrainImgLoader = torch.utils.data.DataLoader(
        # YH: Supply an instance of 'DA.myImageFloder' which indicates a "dataset object" to load data from.
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    # YH: Setup the logger path and name.
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = logger.setup_logger(args.save_path + '/training.log')
    # YH: Save command line arguments in log.
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    # YH: Specify the model to be "AnyNet" and supply command line arguments as parameter for its initialization.
    model = models.anynet.AnyNet(args)
    # YH: Data parallelism at the module level.
    model = nn.DataParallel(model).cuda()
    '''
    YH: The 'optim' package in PyTorch abstracts the idea of an optimization algorithm 
        and provides implementations of commonly used optimization algorithms.
    '''
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # YH: "model.parameters" returns an iterator over module parameters, which is different to "state_dict".
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    '''
    YH: When it comes to saving and loading models, there are three core functions to be familiar with:
        1. torch.save: Saves a serialized object to disk. This function uses Python’s "pickle" utility for serialization. 
           Models, tensors, and dictionaries of all kinds of objects can be saved using this function.
        2. torch.load: Uses "pickle"’s unpickling facilities to deserialize pickled object files to memory. 
           This function also facilitates the device to load the data into.
        3. torch.nn.Module.load_state_dict: Loads a model’s parameter dictionary using a deserialized "state_dict". 
        
        In PyTorch, the learnable parameters (i.e. weights and biases) of an "torch.nn.Module" model 
        are contained in the model’s parameters (accessed with "model.parameters()"). 
        A "state_dict" is simply a Python dictionary object that maps each layer to its parameter tensor. 
        Note that only layers with learnable parameters (convolutional layers, linear layers, etc.) 
        and registered buffers (batchnorm’s running_mean) have entries in the model’s "state_dict". 
        Optimizer objects ("torch.optim") also have a "state_dict", which contains information 
        about the optimizer’s state, as well as the hyperparameters used.
    '''
    args.start_epoch = 0
    if args.resume:
        # YH: If it is a resuming task, then load a file with the name contained in "args.resume".
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            # YH: Load the checkpoint from the file with the name contained in "args.resume".
            checkpoint = torch.load(args.resume)
            # YH: Load the current epoch
            args.start_epoch = checkpoint['epoch']
            # YH: Load the model state from the checkpoint
            model.load_state_dict(checkpoint['state_dict'])
            # YH: Load the optimizer state from the checkpoint
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')

    start_full_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
            
        # YH: For each epoch, execute "train()" procedure for model 'AnyNet'.
        train(TrainImgLoader, model, optimizer, log, epoch)

        # YH: After a training, store the updates by "torch.save" into the file "checkpoint.tar" which is a common PyTorch convention.
        savefilename = args.save_path + '/checkpoint.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, savefilename)

    # YH: After all training epochs, perform testing
    test(TestImgLoader, model, log)
    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))

# YH: An epoch training contains several batches of examples, each batch runs a forward-loss-backward-optimize.
def train(dataloader, model, optimizer, log, epoch=0):

    # YH: We can choose with training with spn or not
    stages = 3 + args.with_spn
    
    # YH: Initiate a list of AverageMeter instances for each stage
    losses = [AverageMeter() for _ in range(stages)]
    
    # YH: I guess the length of a dataloader is the # of training examples???
    length_loader = len(dataloader)

    # YH: Set the model in "train mode"
    model.train()
    
    '''
    YH: The loop is the place where the training really happens.
        'dataloader' is 'torch.utils.data.DataLoader', which is an iterator.
        An epoch divides all examples into several batches, each batch contains a bunch of examples.
        An epoch will go through all examples.
        A batch iteration will execute a forward propagation with the batch's examples,
        average the loss and use Gradient Descent for a backword propagation.
    '''
    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        # YH: Ground truth
        disp_L = disp_L.float().cuda()

        '''
        YH: Set the gradients of all optimized torch.Tensors to zero.
            In PyTorch, we need to set the gradients to zero before kicking off backpropragation 
            because PyTorch accumulates the gradients on subsequent backward passes. 
            This is convenient while training RNNs. 
            
            So, the default action is to accumulate (i.e. sum) the gradients on every 'loss.backward()' call.
            Because of this, when you start your training loop, 
            ideally you should zero out the gradients so that you do the parameter update correctly. 
            Else the gradient would point in some other direction 
            than the intended direction towards the minimum (or maximum, in case of maximization objectives).
        '''
        optimizer.zero_grad()
        
        # YH: Create a mask with the shape of matrix "disp_L", where "true" for "disp_L < args.maxdisp" and "false" otherwise.
        #     That is, only those disparity value less than the 'maxdisp' will be preserved.
        mask = disp_L < args.maxdisp
        if mask.float().sum() == 0:
            continue
        '''  
        YH: detach() doesn't affect the original graph. 
            It will create a copy of the variable with "requires_grad = false",
            and returns a new Variable separated from the current image.
            The returned Variable will never need a gradient.
            I guess it is saying that mask is created by other variables, 
            but after detach, it will not change along with those variable changings.
        '''
        mask.detach_()
        
        # YH: Run the model as an initiated AnyNet, on the batch of examples, which will call the model's "forward" method.
        outputs = model(imgL, imgR)
        
        # YH: Returns a tensor with all the dimensions of input of size 1 removed.
        outputs = [torch.squeeze(output, 1) for output in outputs]
        
        '''
        YH: Compute loss in each stage.
            "F.smooth_l1_loss" creates a criterion that 
            uses a 'squared term' if the absolute element-wise error falls below beta 
            and an L1 term otherwise.
            It is less sensitive to outliers than the "MSELoss" 
            and in some cases prevents exploding gradients (e.g. see Fast R-CNN paper by Ross Girshick). 
            Also known as the "Huber loss".
        '''
        # YH: For every stage in this batch, it calculates the disparity difference only for disparity value < maxdisparity.
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(stages)]
        
        # YH: Backward propagation (compute the gradient) after computing loss
        sum(loss).backward()
        
        # YH: It updates the learnable parameters with Gradient and Learning rate for next batch iteration.
        #     This is a simplified version supported by most optimizers. 
        #     The function can be called once the gradients are computed using e.g. backward().
        optimizer.step()

        # YH: Compute losses by 'AverageMeter' for each stage in this batch for using in log.
        for idx in range(stages):
            losses[idx].update(loss[idx].item()/args.loss_weights[idx])

        # YH: It only logs for every fixed number of batches
        if batch_idx % args.print_freq:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(stages)]
            info_str = '\t'.join(info_str)
            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))
    
    # YH: Log the final loss in each stage for this epoch.
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)


def test(dataloader, model, log):

    stages = 3 + args.with_spn
    EPEs = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)
    
    # YH: Set the model in "evaluation mode"
    model.eval()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        # YH: Ground truth
        disp_L = disp_L.float().cuda()

        mask = disp_L < args.maxdisp
        
        '''
        YH: Inference has only forward propagation and no backward propagation.
            Disabling gradient calculation is useful for inference, 
            when you are sure that you will not call Tensor.backward(). 
            It will reduce memory consumption for computations that would otherwise have requires_grad=True.
        '''
        with torch.no_grad():
            # YH: Inference
            outputs = model(imgL, imgR)
            for x in range(stages):
                if len(disp_L[mask]) == 0:
                    EPEs[x].update(0)
                    continue
                output = torch.squeeze(outputs[x], 1)
                output = output[:, 4:, :]
                # YH: For the current stage, calculate the Disparity difference using the absolute mean, and update the average difference of batches so far.
                EPEs[x].update((output[mask] - disp_L[mask]).abs().mean())

        # YH: For every batch we compute loss of individual stage and log them.
        info_str = '\t'.join(['Stage {} = {:.2f}({:.2f})'.format(x, EPEs[x].val, EPEs[x].avg) for x in range(stages)])
        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))
            
    # YH: Log the final loss in each stage for this evaluation procedure
    info_str = ', '.join(['Stage {}={:.2f}'.format(x, EPEs[x].avg) for x in range(stages)])
    log.info('Average test EPE = ' + info_str)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
'''
YH: Python defines a special variable called "__name__" that contains a string whose value depends on how the code is being used.
    __name__ is stored in the global namespace of the module along with the __doc__, __package__, and other attributes.

    Execution Modes in Python
    There are two primary ways that you can instruct the Python interpreter to execute or use code:
        1. You can execute the Python file as a script using the command line.
           In this case, __name__ == '__main__'
        2. You can import the code from one Python file into another file or into the interactive interpreter.
           When the Python interpreter imports code, the value of __name__ is set to be the same as the name of the module that is being imported.
'''
    main()
