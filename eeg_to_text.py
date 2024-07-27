import os
import torch
import torch.nn as nn
import datetime
import numpy as np
import pickle
import torch.optim as optim
import pdb
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from data import EEGWordDataset
from my_model import EEGToWord

def train(model, dataloaders, device, num_epochs, optimizer, criterion, scheduler, checkpoint_path_best = './checkpoints/decoding/best/temp_decoding.pt', checkpoint_path_last = './checkpoints/decoding/last/temp_decoding.pt'):
    best_loss = 100000000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            epoch_loss_total = 0.0
            count=0
            for word_eeg, word_id, attention_mask in tqdm(dataloaders[phase]):
                word_eeg = word_eeg.to(device).float()
                word_id = word_id.to(device)
                attention_mask = attention_mask.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(word_eeg, word_id)
                    loss = criterion(output, word_id, attention_mask)
                    if phase == 'train':
                        loss.sum().backward()
                        optimizer.step()
                    batch_loss = loss.sum()
                    epoch_loss_total += batch_loss
                    running_loss += batch_loss
            if phase == 'train':
                scheduler.step()
            epoch_loss = epoch_loss_total / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                '''save checkpoint'''
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'update best on dev checkpoint: {checkpoint_path_best}')
    
    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')

    return model

def eval(model, dataloaders, device, tokenizer, output_path='./results/temp.txt'):
    model.eval()
    acc_count = 0
    with open(output_path,'w') as f:
        for word_eeg, word_id, attention_mask in tqdm(dataloaders['test']):
            word_eeg = word_eeg.to(device).float()
            word_id = word_id.to(device)
            attention_mask = attention_mask.to(device)
            output = model(word_eeg, word_id)

            target_word = tokenizer.decode(word_id[0], skip_special_tokens = True)
            f.write(f'target word: {target_word}\t')
            valid_len = get_valid_len(attention_mask)[0]
            pred_ids = output.argmax(dim=2)[0][:valid_len]
            pred_word = tokenizer.decode(pred_ids, skip_special_tokens = True)
            f.write(f'pred word: {pred_word}\n')
            if pred_word == target_word:
                acc_count +=1
    f.close()
    accuracy = acc_count / len(dataloaders['test'].dataset)
    print('accuracy: {:4f}'.format(accuracy))


def get_valid_len(attention_mask):
    bool_tensor = attention_mask.bool()
    valid_len = torch.sum(bool_tensor, dim=1)
    return valid_len

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, attention_mask):
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * attention_mask).sum(dim=1)
        valid_len = get_valid_len(attention_mask)
        loss = weighted_loss / valid_len
        return loss

if __name__ == '__main__':
    num_epochs = 30
    batch_size = 128
    lr = 1e-5
    phase = 'test'

    save_path = './checkpoints/word-level'
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

    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:7" 
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
    
    train_set = EEGWordDataset(input_dataset_dicts=whole_dataset_dicts, phase='train', tokenizer=tokenizer)
    dev_set = EEGWordDataset(input_dataset_dicts=whole_dataset_dicts, phase = 'dev', tokenizer=tokenizer)
    test_set = EEGWordDataset(input_dataset_dicts=whole_dataset_dicts, phase = 'test', tokenizer=tokenizer)
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    dev_dataloader = DataLoader(dev_set, batch_size = 1, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size = 1, num_workers=4)
    dataloaders = {'train':train_dataloader, 'dev':dev_dataloader, 'test':test_dataloader}

    ''' set up model '''
    model = EEGToWord(target_vocab_size=tokenizer.vocab_size)
    if phase == 'test':
         model.load_state_dict(torch.load('./checkpoints/word-level/best/b128_epoch30_lr1e-05_7-26-16-16.pt'))
         model.to(device)
         res_path = os.path.join('./results', f'{save_name}.txt')
         eval(model, dataloaders, device, tokenizer, res_path)
    else:
        model.to(device)

        ''' set up optimizer and scheduler'''
        optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
        criterion = MaskedSoftmaxCELoss()
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)

        trained_model = train(model, dataloaders, device, num_epochs, optimizer, criterion, exp_lr_scheduler, checkpoint_best, checkpoint_last)