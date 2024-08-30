import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import numpy as np
import pickle
import torch.optim as optim
import pdb
import json
import argparse
import wandb

from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from zhipuai import ZhipuAI

from data import EEGWordDataset, EEGSentDataset
from my_model import EEGToWord

wandb.login()
API_KEY = "a21332b650e647890eb4ec3ad9557152.8zM2MDTDliFA9XKM"
with open('vocab.json', 'r', encoding='utf-8') as f:
    # 加载JSON数据到字典
    vocab_dict = json.load(f)

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

            epoch_loss_total = 0.0

            for word_eeg, word_id in tqdm(dataloaders[phase]):
                word_eeg = word_eeg.to(device).float()
                word_id = word_id.to(device)
                # attention_mask = attention_mask.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(word_eeg)
                    loss = criterion(output, word_id)
                    # loss = criterion(output, word_id, attention_mask)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    batch_loss = loss.item()*word_eeg.shape[0]
                    epoch_loss_total += batch_loss
                if phase=='train':
                    wandb.log({"Loss/train": loss.item()})
                else:
                    wandb.log({"Loss/dev": loss.item()})
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
        for word_eeg, word_id in tqdm(dataloaders['test']):
            word_eeg = word_eeg.to(device).float()
            word_id = word_id
            # attention_mask = attention_mask.to(device)
            output = model(word_eeg)

            # target_word = tokenizer.decode(word_id[0], skip_special_tokens = True)
            for key, val in vocab_dict.items():
                if val == word_id:
                    target_word = key
                    break
            f.write(f'target word: {target_word}\t')
            # pred_word = get_pred_word(output, tokenizer, attention_mask)
            probabilities = F.softmax(output,dim=1)
            pred_id = probabilities.argmax(dim=1)[0]
            for key, val in vocab_dict.items():
                if val == pred_id:
                    pred_word = key
                    break
            f.write(f'pred word: {pred_word}\n')
            if pred_word == target_word:
                acc_count +=1
    f.close()
    accuracy = acc_count / len(dataloaders['test'].dataset)
    print('accuracy: {:4f}'.format(accuracy))

def get_pred_word(model_output, tokenizer, attention_mask):
    valid_len = get_valid_len(attention_mask)[0]
    pred_ids = model_output.argmax(dim=2)[0][:valid_len]
    pred_word = tokenizer.decode(pred_ids, skip_special_tokens = True)
    return pred_word

def generate_sent(model, dataloaders, device, tokenizer, output_path='./results/temp.txt'):
    target_word_list = []
    target_sent_list = []
    pred_word_list = []
    pred_sent_list = []
    with open(output_path,'w') as f:
        for ground_truth, sent_eeg, target_id, target_word, attention_mask in tqdm(dataloaders['test_sent']):
            target_words = []
            # target_words = ground_truth[0].split()
            pred_words = []
            for i in range(len(attention_mask)):
                if torch.all(attention_mask[i] == 0):
                    break
                word_eeg = sent_eeg[i].to(device).float()
                word_id = target_id[i].to(device)
                word_attention_mask = attention_mask[i].to(device)
                output = model(word_eeg, word_id)

                pred_word = get_pred_word(output, tokenizer, word_attention_mask)
                target_words.append(target_word[i][0])
                pred_words.append(pred_word)
                
            target_word_list.append([target_words])
            pred_word_list.append(pred_words)
            # target_sent = ' '.join(target_words)
            target_sent = ground_truth[0]
            target_sent_list.append(target_sent)
            pred_sent = ' '.join(pred_words)
            pred_sent_list.append(pred_sent)
            refined_sent = refine_sent(pred_sent)
            f.write(f'target sentence: {target_sent}\n')
            # f.write(f'predicted sentence: {pred_sent}\n')
            f.write(f'predicted sentence: {refined_sent}\n')
            f.write('################################################\n\n\n')
    f.close()

    """ calculate corpus bleu score """
    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
    for weight in weights_list:
        corpus_bleu_score = corpus_bleu(target_word_list, pred_word_list, weights = weight)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)

    print()
    """ calculate rouge score """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_sent_list, target_sent_list, avg = True)
    print(rouge_scores)

def refine_sent(src_sent):
    client = ZhipuAI(api_key=API_KEY)
    prompt = f'As a text reconstructor, your task is to restore corrupted sentences to their original form while making minimum changes. You should adjust the spaces and punctuation marks as necessary. Do not introduce any additional information. Return only the refactored sentence, do not return any additional information. Reconstruct the following text: [{src_sent}].'
    response = client.chat.completions.create(
        model="glm-4-0520",  # 填写需要调用的模型编码
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content

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
    parser = argparse.ArgumentParser(description='Specify config args for decoding')
    parser.add_argument('--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', default = 'cuda:0')
    parser.add_argument('--num_epochs', type = int, help='num_epochs', default = 30, required=True)
    parser.add_argument('--batch_size', type = int, help='batch_size', default = 32, required=True)
    parser.add_argument('--lr', type = float, help='learning_rate', default = 0.00005, required=True)
    parser.add_argument('--phase', help='train phase or test phase', default = 'train', required=True)
    parser.add_argument('--checkpoint_path', help='checkpoint_path', default = './checkpoints/word-level/best/b32_epoch50_lr1e-05_8-23-10-27.pt')
    args = vars(parser.parse_args())

    num_epochs = args['num_epochs']
    batch_size = args['batch_size']
    lr = args['lr']
    phase = args['phase']
    checkpoint_path = args['checkpoint_path']

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
        dev = args['cuda']
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')

    '''init wandb'''
    run = wandb.init(
        # Set the project where this run will be logged
        project="EEG-To-Word",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": num_epochs,
        },
    )

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
    test_sent_set = EEGSentDataset(input_dataset_dicts=whole_dataset_dicts, phase = 'test', tokenizer=tokenizer)

    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    dev_dataloader = DataLoader(dev_set, batch_size = 1, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size = 1, num_workers=4)
    test_sent_dataloader = DataLoader(test_sent_set, batch_size = 1, num_workers=4)
    dataloaders = {'train':train_dataloader, 'dev':dev_dataloader, 'test':test_dataloader, 'test_sent':test_sent_dataloader}

    ''' set up model '''
    # model = EEGToWord(vocab_size=tokenizer.vocab_size)
    model = EEGToWord(vocab_size=len(vocab_dict))
    if phase == 'test':
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(device)
        res_path = os.path.join('./results/word-level', f'{save_name}.txt')
        eval(model, dataloaders, device, tokenizer, res_path)
        # if level == 'word':
        #     res_path = os.path.join('./results/word-level', f'{save_name}.txt')
        #     eval(model, dataloaders, device, tokenizer, res_path)
        # elif level == 'sentence':
        #     res_path = os.path.join('./results/sent-level', f'{save_name}.txt')
        #     generate_sent(model, dataloaders, device, tokenizer, res_path)
    else:
        model.to(device)

        ''' set up optimizer and scheduler'''
        optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
        # criterion = MaskedSoftmaxCELoss()
        criterion = nn.CrossEntropyLoss()
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)

        trained_model = train(model, dataloaders, device, num_epochs, optimizer, criterion, exp_lr_scheduler, checkpoint_best, checkpoint_last)