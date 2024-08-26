import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import datetime
import numpy as np
import pickle
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BartTokenizer
import pdb
import wandb

from data import ZuCo_dataset
from pretrain_model import EEGAutoencoder

wandb.login()

def train_model(dataset, device, batch_size, model, criterion, optimizer, scheduler, num_epochs, checkpoint_path_best = './checkpoints/pretrain/best/temp_decoding.pt', checkpoint_path_last = './checkpoints/pretrain/last/temp_decoding.pt'):
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000

    for epoch in range(num_epochs):
        # 计算需要掩码的元素数量
        mask_ratio = 0.15
        for EEGObj in dataset.inputs:
            total_elements = EEGObj['seq_len'] * EEGObj['input_embeddings'].shape[1]  # 计算张量中元素的总数
            mask_count = int(total_elements * mask_ratio)

            # 生成掩码索引
            mask_indices = torch.randperm(total_elements)[:mask_count]  # 随机排列所有索引，然后取前mask_count个

            # 将掩码索引对应的元素设置为0
            rawEEG_flat = EEGObj['input_embeddings'].reshape(-1)  # 将张量展平为一维
            rawEEG_flat[mask_indices] = 0  # 将掩码索引对应的元素设置为0
            EEGObj['maskedData'] = rawEEG_flat.reshape(EEGObj['input_embeddings'].size())  # 恢复原始形状
        train_dataloder=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=4)

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()
        running_loss = 0.0
        epoch_loss_total = 0.0
        count=0
        
        for rawData, maskedData, input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(train_dataloder):
            count+=1
            origin_EEG = input_embeddings.to(device).float()
            masked_EEG = maskedData.to(device).float()
            input_mask_invert = input_mask_invert.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            recon_EEG = model(masked_EEG, input_mask_invert)
            loss = criterion(origin_EEG, recon_EEG)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item() * origin_EEG.size()[0] # batch loss
            epoch_loss_total += batch_loss
            running_loss += batch_loss
            if (count * origin_EEG.size()[0])%128 == 0:
                wandb.log({"Loss/pretrain": running_loss/128})
                running_loss=0.0

        scheduler.step()
        epoch_loss = epoch_loss_total / len(dataset)
        print('{} Loss: {:.4f}'.format('pretrain', epoch_loss))

        # deep copy the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # best_model_wts = copy.deepcopy(model.state_dict())
            '''save checkpoint'''
            torch.save(model.state_dict(), checkpoint_path_best)
            print(f'update best on dev checkpoint: {checkpoint_path_best}')
            print()

    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')

    return model
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify config args for pretraining')
    parser.add_argument('--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', default = 'cuda:0')
    parser.add_argument('--num_epochs', type = int, help='num_epochs', default = 30, required=True)
    parser.add_argument('--batch_size', type = int, help='batch_size', default = 32, required=True)
    parser.add_argument('--lr', type = float, help='learning_rate', default = 0.00005, required=True)
    args = vars(parser.parse_args())
    
    num_epochs = args['num_epochs']
    batch_size = args['batch_size']
    lr = args['lr']

    save_path = './checkpoints/pretrain'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    current_datetime = datetime.datetime.now()
    save_name = f'b{batch_size}_epoch{num_epochs}_lr{lr}_{current_datetime.month}-{current_datetime.day}-{current_datetime.hour}-{current_datetime.minute}'
    save_path_best = os.path.join(save_path, 'best')
    if not os.path.exists(save_path_best):
        os.makedirs(save_path_best)
    checkpoint_best = os.path.join(save_path_best, f'{save_name}.pt')
    save_path_last = os.path.join(save_path, 'last')
    if not os.path.exists(save_path_last):
        os.makedirs(save_path_last)
    checkpoint_last = os.path.join(save_path_last, f'{save_name}.pt')

    '''init wandb'''
    run = wandb.init(
        # Set the project where this run will be logged
        project="EEG-To-Text",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": num_epochs,
        },
    )

    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')

    ''' set up dataloader '''
    whole_dataset_dicts = []
    # task1
    with open('./dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle', 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    # task2
    with open('./dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle', 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    # task2.2
    with open('./dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle', 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    tokenizer = BartTokenizer.from_pretrained('bart-large')
    # train dataset
    train_set = ZuCo_dataset(input_dataset_dicts = whole_dataset_dicts, phase = 'pretrain', tokenizer = tokenizer)
    print('[INFO]train_set size: ', len(train_set))
    # train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)

    ''' set up model '''
    model = EEGAutoencoder()
    model.to(device)

    ''' set up optimizer and scheduler'''
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)

    ''' set up loss function '''
    criterion = nn.MSELoss()

    trained_model = train_model(train_set, device, batch_size, model, criterion, optimizer, scheduler=exp_lr_scheduler, num_epochs=num_epochs, checkpoint_path_best=checkpoint_best, checkpoint_path_last=checkpoint_last)
    
