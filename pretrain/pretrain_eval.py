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
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import pdb
import wandb

from data import ZuCo_dataset
from pretrain_model import BrainTranslator

def eval_model(dataloaders, device, tokenizer, model, output_all_results_path = './results/temp.txt' ):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    model.eval()   # Set model to evaluate mode
    running_loss = 0.0

    # Iterate over data.
    sample_count = 0
    
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    with open(output_all_results_path,'w') as f:
        for rawData, maskedData, input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(dataloaders['test']):
            # load in batch
            input_embeddings_batch = input_embeddings.to(device).float()
            input_masks_batch = input_masks.to(device)
            target_ids_batch = target_ids.to(device)
            input_mask_invert_batch = input_mask_invert.to(device)
            
            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens = True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens = True)
            f.write(f'target string: {target_string}\n')

            # add to list for later calculate bleu metric
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)
            
            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

            # forward
            seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)

            """calculate loss"""
            loss = seq2seqLMoutput.loss # use the BART language modeling loss

            # get predicted tokens
            logits = seq2seqLMoutput.logits # 8*48*50265
            probs = logits[0].softmax(dim = 1)
            values, predictions = probs.topk(1)
            predictions = torch.squeeze(predictions)
            predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>','')
            f.write(f'predicted string: {predicted_string}\n')
            f.write('################################################\n\n\n')

            # convert to int list
            predictions = predictions.tolist()
            truncated_prediction = []
            for t in predictions:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)

            sample_count += 1
            # statistics
            running_loss+=loss.item() * input_embeddings_batch.size()[0]


    epoch_loss = running_loss / dataset_sizes['test_set']
    print('test loss: {:4f}'.format(epoch_loss))

    """ calculate corpus bleu score """
    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
    for weight in weights_list:
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights = weight)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)

    print()
    """ calculate rouge score """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_string_list,target_string_list, avg = True)
    print(rouge_scores)


if __name__ == '__main__':
    batch_size = 1

    results_path = './results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    current_datetime = datetime.datetime.now()
    save_name = f'{current_datetime.month}-{current_datetime.day}-{current_datetime.hour}-{current_datetime.minute}'
    result_name = os.path.join(results_path, f'{save_name}.txt')

    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = 'cuda:3'
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
    test_set = ZuCo_dataset(input_dataset_dicts = whole_dataset_dicts, phase = 'test', tokenizer = tokenizer)
    dataset_sizes = {'test_set': len(test_set)}
    test_dataloader = DataLoader(test_set, batch_size = 1, num_workers=4)
    dataloaders = {'test':test_dataloader}

    ''' set up model '''
    pretrained = BartForConditionalGeneration.from_pretrained('bart-large')
    model = BrainTranslator(pretrained)
    model.load_state_dict(torch.load('./checkpoints/pretrain/decoding/best/b16_epoch30_lr2e-06_7-18-22-27.pt'))
    model.to(device)

    trained_model = eval_model(dataloaders, device, tokenizer, model, result_name)
