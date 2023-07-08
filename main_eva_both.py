from operator import mod
import torch
import random
import numpy as np
import argparse
import os
import sys
import math
import time
import shutil
import pickle
import torch.nn as nn
from classifier.vcop_eva_both import CoOp
from classifier.proda import ProDA
# from dataset.build_dataset import build_dataset, build_dataset_fs
import dataset.evaluation_dataloader_both as incremental_dataloader
from utils import mkdir_p
import pdb


def parse_option():
    parser = argparse.ArgumentParser('Prompt Learning for CLIP', add_help=False)

    parser.add_argument("--root_1", type=str, default='/sda5/cifar100',help='root')
    parser.add_argument("--root_2", type=str, default='/sda5/Imagenet100/ImageNet100',help='root')
    parser.add_argument("--aug",type=str, default='flip', help='root')

    parser.add_argument("--mean_per_class", action='store_true', help='mean_per_class')
    parser.add_argument("--db_name_1", type=str, default='cifar100', help='dataset name')
    parser.add_argument("--db_name_2", type=str, default='imagenet100', help='dataset name')
    parser.add_argument("--num_runs", type=int, default=1, help='num_runs')
    parser.add_argument("--seed", type=int, default=0, help='random seed')

    parser.add_argument("--arch", type=str, default='ViT-L-14', help='arch') #ViT-L-14 is best
    parser.add_argument("--checkpoint", type=str, default='/home/wrq/ProDa/ckpt/', help='save_checkpoint')
    parser.add_argument("--ckpt_path", type=str, default=None, help='ckpt_path')
    parser.add_argument("--keyprompt_path", type=str, default='/home/wrq/ProDa/save/cifar100/vcop_4', help='keyprompt_path')

    # optimization setting
    parser.add_argument("--lr", type=float, default=1e-3, help='num_runs')#1e-3
    parser.add_argument("--wd", type=float, default=0.0, help='num_runs')
    parser.add_argument("--epochs", type=int, default=10, help='num_runs')
    parser.add_argument("--train_batch", type=int, default=16, help='num_runs')#16
    parser.add_argument("--test_batch", type=int, default=8, help='num_runs')#8

    #model setting
    parser.add_argument("--model", type=str, default='coop', help='model')
    parser.add_argument("--n_prompt", type=int, default=32, help='num_runs')
    parser.add_argument("--prompt_bsz", type=int, default=4, help='num_runs')

    #incremental setting
    parser.add_argument("--num_class", type=int, default=200, help='num_class')
    parser.add_argument("--class_per_task", type=int, default=10, help='class per task')
    parser.add_argument("--num_task", type=int, default=20, help='num_task')
    parser.add_argument("--start_sess", type=int, default=0, help='start session')
    parser.add_argument("--sess", type=int, default=0, help='current session')
    parser.add_argument("--memory", type=int, default=1000, help='memory')
    parser.add_argument("--num_test", type=int, default=15, help='num_test_text')
    parser.add_argument("--num_prompt", type=int, default=10, help='num_prompt')
    parser.add_argument("--text_prompt", type=int, default=3, help='text_prompt')
    parser.add_argument("--keep", type=bool, default=True, help='keep')#continue from other datasets

    args, unparsed = parser.parse_known_args()

    args.mean_per_class = False
    # if args.db_name in ['oxfordpets', 'fgvc_aircraft', 'oxford_flowers', 'caltech101']:
    #     args.mean_per_class = True

    # if args.db_name in ['cifar10', 'eurosat','stl1e']:
    #     args.num_runs= 10
    # # args.num_runs = 1

    if args.ckpt_path is None:
        args.ckpt_path = '/home/wrq/ProDa/pretrain/{}.pt'.format(args.arch)

    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    prev_key, prev_prompt = False, False
    # import pdb;pdb.set_trace()
    setup_seed(args.seed)
    if args.keep==True:
        # pdb.set_trace() 
        path_key=os.path.join(args.keyprompt_path, 'text_key.pth.tar')
        path_prompt=os.path.join(args.keyprompt_path, 'text_prompt.pth.tar')
        # pdb.set_trace()  
        prev_key=torch.load(path_key)
        prev_prompt=torch.load(path_prompt)
        # print('prompt trained from previous dataset')
        print('load from:', args.keyprompt_path)
    else:
        print('prompt trained from random')
    if args.model == 'coop':
        model = CoOp(prev_key,prev_prompt,args=args,keep=args.keep)
    elif args.model == 'proda':
        model = ProDA(args=args)
    if not os.path.isdir(args.ckpt_path):
        mkdir_p(args.checkpoint)
    np.save(args.checkpoint + "/seed.npy", args.seed)
    inc_dataset = incremental_dataloader.IncrementalDataset(
                        dataset1_name=args.db_name_1,
                        dataset2_name=args.db_name_2,
                        args = args,
                        random_order=False, #random class
                        shuffle=True,
                        seed=args.seed,
                        batch_size=args.train_batch,
                        workers=8,
                        validation_split=0,
                        increment=args.class_per_task,
                    )      
    start_sess = args.start_sess
    memory = None
    # for ses in range(start_sess, args.num_task):
    ses = args.num_task - 1
    #train_loader: memory+当前train_samples  for_memory：当前task的图片位置和对应标签    
    task_info, train_loader, class_name, test_class, test_loader_1, test_loader_2, for_memory = inc_dataset.new_task(ses, memory) 
    args.sess=ses       
    print('ses:',ses)
    print(task_info)    
    print(inc_dataset.sample_per_task_testing)     # dict{task:len(test)}
    args.sample_per_task_testing = inc_dataset.sample_per_task_testing
    acc = model.accuracy(test_loader_1, test_loader_2, args.num_test, test_class, mean_per_class=args.mean_per_class)
    print('acc',acc)
        
if __name__ == '__main__':
    args = parse_option()
    main(args)