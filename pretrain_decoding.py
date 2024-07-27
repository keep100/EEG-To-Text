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
from transformers import BartTokenizer, BartForConditionalGeneration
import pdb
import wandb

from data import ZuCo_dataset
from pretrain_model import BrainTranslator

wandb.login()

def train_model(dataloaders, device, tokenizer, model, optimizer, scheduler, num_epochs, checkpoint_path_best = './checkpoints/decoding/best/temp_decoding.pt', checkpoint_path_last = './checkpoints/decoding/last/temp_decoding.pt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            epoch_loss_total = 0.0
            count=0

            # Iterate over data.
            for rawData, maskedData, input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(dataloaders[phase]):
                
                count+=1
                # load in batch
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                """replace padding ids in target_ids with -100"""
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
    	        # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)
                    
                    """calculate loss"""
                    loss = seq2seqLMoutput.loss

                    if phase == 'train':
                        # with torch.autograd.detect_anomaly():
                        loss.backward()
                        optimizer.step()

                # statistics
                batch_loss = loss.item() * input_embeddings_batch.size()[0] # batch loss
                epoch_loss_total+=batch_loss
                running_loss+=batch_loss
                if (count*input_embeddings_batch.size()[0])%128 ==0:
                    if phase=='train':
                        wandb.log({"Loss/train": running_loss/128})
                    else:
                        wandb.log({"Loss/dev": running_loss/128})
                    running_loss=0.0
                

            if phase == 'train':
                scheduler.step()

            epoch_loss = epoch_loss_total / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                # best_model_wts = copy.deepcopy(model.state_dict())
                '''save checkpoint'''
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'update best on dev checkpoint: {checkpoint_path_best}')

    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify config args for decoding')
    parser.add_argument('--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', default = 'cuda:0')
    parser.add_argument('--num_epochs', type = int, help='num_epochs', default = 30, required=True)
    parser.add_argument('--batch_size', type = int, help='batch_size', default = 32, required=True)
    parser.add_argument('--lr', type = float, help='learning_rate', default = 0.00005, required=True)
    args = vars(parser.parse_args())
    
    num_epochs = args['num_epochs']
    batch_size = args['batch_size']
    lr = args['lr']

    save_path = './checkpoints/pretrain/decoding'
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
    train_set = ZuCo_dataset(input_dataset_dicts = whole_dataset_dicts, phase = 'train', tokenizer = tokenizer)
    dev_set = ZuCo_dataset(input_dataset_dicts = whole_dataset_dicts, phase = 'dev', tokenizer = tokenizer)
    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    dev_dataloader = DataLoader(dev_set, batch_size = 1, num_workers=4)
    dataloaders = {'train':train_dataloader, 'dev':dev_dataloader}

    ''' set up model '''
    pretrained = BartForConditionalGeneration.from_pretrained('bart-large')
    model = BrainTranslator(pretrained)
    model.load_state_dict(torch.load('./checkpoints/pretrain/best/b32_epoch30_lr5e-05_7-17-15-55.pt'),strict=False)
    model.to(device)

    ''' set up optimizer and scheduler'''
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)

    trained_model = train_model(dataloaders, device, tokenizer, model, optimizer, scheduler=exp_lr_scheduler, num_epochs=num_epochs, checkpoint_path_best=checkpoint_best, checkpoint_path_last=checkpoint_last)
  