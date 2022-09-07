import argparse
import subprocess
import torch
import torchvision
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
from collections import OrderedDict
import wandb
from datetime import datetime
import numpy as np

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
        print(logline)
        f.write(logline + '\n')
        
device = torch.device('cuda')

if __name__ == '__main__':
    
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    wandb.init(project='showroom top pose estimation',  name=now, mode='online', dir='./')
#    wandb.run.log_code('./coco.py')
#    wandb.run.log_code('./configs')
    config = wandb.config

    
    print('Loading config %s' % args.config)
    with open(args.config, 'r') as f:
        config = json.load(f)
        pprint.pprint(config)
        
    logfile_path = 'logs/' + args.config.split('/')[-1] + '.log'
    checkpoint_dir = 'checkpoints/' + args.config.split('/')[-1] + '.checkpoints'

    if not os.path.exists(checkpoint_dir):
        print('Creating checkpoint directory % s' % checkpoint_dir)
        os.mkdir(checkpoint_dir)
    
        
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
    
    part_type_counts = test_dataset.get_part_type_counts().float().cuda()
    part_weight = 1.0 / part_type_counts
    part_weight = part_weight / torch.sum(part_weight)
    paf_type_counts = test_dataset.get_paf_type_counts().float().cuda()
    paf_weight = 1.0 / paf_type_counts
    paf_weight = paf_weight / torch.sum(paf_weight)
    paf_weight /= 2.0
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        **config["train_loader"]
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        **config["test_loader"]
    )
    
    model = MODELS[config['model']['name']](**config['model']['kwargs']).to(device)
    wandb.watch(model, log='all')
    
    if "initial_state_dict" in config['model']:
        print('Loading initial weights from %s' % config['model']['initial_state_dict'])
        model.load_state_dict(torch.load(config['model']['initial_state_dict']))
    
    optimizer = OPTIMIZERS[config['optimizer']['name']](model.parameters(), **config['optimizer']['kwargs'])
    model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
    
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
            save_checkpoint(model, checkpoint_dir, epoch)
        
           
        
        train_loss = 0.0
        model = model.train()
        for image, cmap, paf, mask in tqdm.tqdm(iter(train_loader)):
            image = image.to(device)
            cmap = cmap.to(device)
            paf = paf.to(device)
            
            if mask_unlabeled:
                mask = mask.to(device).float()
            else:
                mask = torch.ones_like(mask).to(device).float()
            
            optimizer.zero_grad()
            cmap_out, paf_out = model(image)
            
            cmap_mse = torch.mean(mask * (cmap_out - cmap)**2)
            paf_mse = torch.mean(mask * (paf_out - paf)**2)
            
            loss = cmap_mse + paf_mse
            wandb.log({'train_loss':loss.item()})
            wandb.log({'train_cmap_loss':cmap_mse.item()})
            wandb.log({'train_paf_loss':paf_mse.item()})
            wandb.log({'epoch':epoch})
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
#             loss.backward()
            optimizer.step()
            train_loss += float(loss)
            
        train_loss /= len(train_loader)
        
        test_loss = 0.0
        model = model.eval()
        for image, cmap, paf, mask in tqdm.tqdm(iter(test_loader)):
      
            with torch.no_grad():
                image = image.to(device)
                cmap = cmap.to(device)
                paf = paf.to(device)
                mask = mask.to(device).float()

                if mask_unlabeled:
                    mask = mask.to(device).float()
                else:
                    mask = torch.ones_like(mask).to(device).float()
                
                cmap_out, paf_out = model(image)
                
                cmap_mse = torch.mean(mask * (cmap_out - cmap)**2)
                paf_mse = torch.mean(mask * (paf_out - paf)**2)

                loss = cmap_mse + paf_mse
                wandb.log({'test_loss':loss.item()})
                wandb.log({'test_cmap_loss':cmap_mse.item()})
                wandb.log({'test_paf_loss':paf_mse.item()})

                test_loss += float(loss)
        test_loss /= len(test_loader)
        
        write_log_entry(logfile_path, epoch, train_loss, test_loss)
        
        
        if 'evaluation' in config:
            evaluator.evaluate(model, train_dataset.topology)
