import argparse
import subprocess
import os
import torch.optim
import tqdm
import apex.amp as amp
import time
import json
import pprint
import torch.nn.functional as F
from coco import CocoDataset, CocoHumanPoseEval
from models import MODELS

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

OPTIMIZERS = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam
}

EPS = 1e-6

def set_lr(optimizer, lr):
    for p in optimizer.param_groups:
        p['lr'] = lr
        
        
def save_checkpoint(model, directory, epoch):
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = os.path.join(directory, 'epoch_%d.pth' % epoch)
    print('Saving checkpoint to %s' % filename)
    torch.save(model.state_dict(), filename)

    
def write_log_entry(logfile, epoch, train_loss, test_loss):
    with open(logfile, 'a+') as f:
        logline = '%d, %f, %f' % (epoch, train_loss, test_loss)
        f.write(logline + '\n')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://192.168.0.93:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', type=bool, default=True)

parser.add_argument('config')

def main():
        
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    args.logfile_path = 'logs/' + args.config.split('/')[-1] + '.log'
    args.checkpoint_dir = 'checkpoints/' + args.config.split('/')[-1] + '.checkpoints'

    if not os.path.exists(args.checkpoint_dir):
        print('Creating checkpoint directory % s' % args.checkpoint_dir)
        os.mkdir(args.checkpoint_dir)
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
        
        
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))   
        
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            # args.randk = gpu
            
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)        

    print('Loading config %s' % args.config)
    with open(args.config, 'r') as f:
        config = json.load(f)
        pprint.pprint(config)
        
    args.batch_size = config["train_loader"]["batch_size"]
    args.workers = config["train_loader"]["num_workers"]
        
    print('model was created')
    model = MODELS[config['model']['name']](**config['model']['kwargs'])
    # print('*****', next(model.parameters()).device)
    
    if "initial_state_dict" in config['model']:
        print('Loading initial weights from %s' % config['model']['initial_state_dict'])
        model.load_state_dict(torch.load(config['model']['initial_state_dict'], map_location=torch.device('cpu')))
             
    optimizer = OPTIMIZERS[config['optimizer']['name']](model.parameters(), **config['optimizer']['kwargs'])  
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')           
       
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                config["train_loader"]["batch_size"] = args.batch_size
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model, optimzier = amp.initialize(model, optimizer, opt_level="O1")
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()   
        
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")                     
        
    # LOAD DATASETS    
    train_dataset_kwargs = config["train_dataset"]
    train_dataset_kwargs['transforms'] = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(**config['color_jitter']),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset_kwargs = config["test_dataset"]
    test_dataset_kwargs['transforms'] = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if 'evaluation' in config:
        evaluator = CocoHumanPoseEval(**config['evaluation'])
    
    train_dataset = CocoDataset(**train_dataset_kwargs)
    test_dataset = CocoDataset(**test_dataset_kwargs)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        test_sampler = None    
    
    part_type_counts = test_dataset.get_part_type_counts().float().to(device)
    part_weight = 1.0 / part_type_counts
    part_weight = part_weight / torch.sum(part_weight)
    paf_type_counts = test_dataset.get_paf_type_counts().float().to(device)
    paf_weight = 1.0 / paf_type_counts
    paf_weight = paf_weight / torch.sum(paf_weight)
    paf_weight /= 2.0
    
    config["train_loader"]['shuffle'] = shuffle=(train_sampler is None)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler,
        **config["train_loader"])
    
    config["test_loader"]['shuffle'] = shuffle=(test_sampler is None)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, sampler=test_sampler,
        **config["test_loader"]
    )
    
    if 'mask_unlabeled' in config and config['mask_unlabeled']:
        print('Masking unlabeled annotations')
        mask_unlabeled = True
    else:
        mask_unlabeled = False

        
    for epoch in range(config["epochs"]):
        
        if str(epoch) in config['stdev_schedule']:
            stdev = config['stdev_schedule'][str(epoch)]
            print('Adjusting stdev to %f' % stdev)
            train_dataset.stdev = stdev
            test_dataset.stdev = stdev
            
        if str(epoch) in config['lr_schedule']:
            new_lr = config['lr_schedule'][str(epoch)]
            print('Adjusting learning rate to %f' % new_lr)
            set_lr(optimizer, new_lr)
        
        if epoch % config['checkpoints']['interval'] == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
                save_checkpoint(model, args.checkpoint_dir, epoch)
        
        
        train_loss = 0.0
        model = model.train()
        # torch.backends.cudnn.benchmark = True
        for image, cmap, paf, mask in tqdm.tqdm(iter(train_loader)):
            
            image = image.to(device, non_blocking=True)
            cmap = cmap.to(device, non_blocking=True)
            paf = paf.to(device, non_blocking=True)
            
            if mask_unlabeled:
                mask = mask.to(device).float()
            else:
                mask = torch.ones_like(mask).to(device).float()
            
            # optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)
            cmap_out, paf_out = model(image)
            
            cmap_mse = torch.mean(mask * (cmap_out - cmap)**2)
            paf_mse = torch.mean(mask * (paf_out - paf)**2)
            
            loss = cmap_mse + paf_mse
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
#             loss.backward()
            optimizer.step()
            train_loss += float(loss)
            
        train_loss /= len(train_loader)
        
        
#         test_loss = 0.0
#         model = model.eval()
#         for image, cmap, paf, mask in tqdm.tqdm(iter(test_loader)):
      
#             with torch.no_grad():
#                 image = image.to(device)
#                 cmap = cmap.to(device)
#                 paf = paf.to(device)
#                 mask = mask.to(device).float()

#                 if mask_unlabeled:
#                     mask = mask.to(device).float()
#                 else:
#                     mask = torch.ones_like(mask).to(device).float()
                
#                 cmap_out, paf_out = model(image)
                
#                 cmap_mse = torch.mean(mask * (cmap_out - cmap)**2)
#                 paf_mse = torch.mean(mask * (paf_out - paf)**2)

#                 loss = cmap_mse + paf_mse

#                 test_loss += float(loss)
#         test_loss /= len(test_loader)
        
#         write_log_entry(logfile_path, epoch, train_loss, test_loss)
        
        
#         if 'evaluation' in config:
#             evaluator.evaluate(model, train_dataset.topology)

if __name__ == '__main__':
    main()
