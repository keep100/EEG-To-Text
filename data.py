import os
import pdb
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import json
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer, BertTokenizer
from tqdm import tqdm
from fuzzy_match import match, algorithims


def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean)/std
    # min_value = torch.min(input_tensor)
    # max_value = torch.max(input_tensor)
    # input_tensor = (input_tensor - min_value) / (max_value - min_value)

    return input_tensor 

def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != 105*len(bands):
            print(f'expect word eeg embedding dim to be {105*len(bands)}, but got {len(word_eeg_embedding)}, return None')
            return None
        if np.isnan(word_eeg_embedding).any():
            return None
        # assert len(word_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)

def get_input_sample(sent_obj, tokenizer, eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], max_len = 56, add_CLS_token = False):
    
    def get_sent_eeg(sent_obj, bands):
        sent_eeg_features = []
        for band in bands:
            key = 'mean'+band
            sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

    if sent_obj is None:
        # print(f'  - skip bad sentence')   
        return None

    input_sample = {}
    # get target label
    target_string = sent_obj['content']
    # if not isinstance(sent_obj['rawData'],np.ndarray):
    #     return None
    # input_sample['rawData']=normalize_1d(torch.from_numpy(sent_obj['rawData']))
    # if input_sample['rawData'].shape[1]<10000:
    #     input_sample['rawData']=F.pad(input_sample['rawData'],(0,10000-input_sample['rawData'].shape[1],0,0))
    # else:
    #     input_sample['rawData']=input_sample['rawData'][:,:10000]

    # input_sample['rawData']={}
    # input_sample['maskedData']={}
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask = True)
    
    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    
    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        # print('[NaN sent level eeg]: ', target_string)
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    # get sentiment label
    # handle some wierd case
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty','empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1','film.')
    
    #if target_string in ZUCO_SENTIMENT_LABELS:
    #    input_sample['sentiment_label'] = torch.tensor(ZUCO_SENTIMENT_LABELS[target_string]+1) # 0:Negative, 1:Neutral, 2:Positive
    #else:
    #    input_sample['sentiment_label'] = torch.tensor(-100) # dummy value
    input_sample['sentiment_label'] = torch.tensor(-100) # dummy value

    # get input embeddings
    word_embeddings = []

    """add CLS token embedding at the front"""
    if add_CLS_token:
        word_embeddings.append(torch.ones(105*len(bands)))

    for word in sent_obj['word']:
        # add each word's EEG embedding as Tensors
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands = bands)
        # check none, for v2 dataset
        if word_level_eeg_tensor is None:
            return None
        # check nan:
        if torch.isnan(word_level_eeg_tensor).any():
            # print()
            # print('[NaN ERROR] problem sent:',sent_obj['content'])
            # print('[NaN ERROR] problem word:',word['content'])
            # print('[NaN ERROR] problem word feature:',word_level_eeg_tensor)
            # print()
            return None
            

        word_embeddings.append(word_level_eeg_tensor)
    # pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))

    input_sample['input_embeddings'] = torch.stack(word_embeddings) # max_len * (105*num_bands)

    # mask out padding tokens
    input_sample['input_attn_mask'] = torch.zeros(max_len) # 0 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask'][:len(sent_obj['word'])+1] = torch.ones(len(sent_obj['word'])+1) # 1 is not masked
    else:
        input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word'])) # 1 is not masked
    

    # mask out padding tokens reverted: handle different use case: this is for pytorch transformers
    input_sample['input_attn_mask_invert'] = torch.ones(max_len) # 1 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])+1] = torch.zeros(len(sent_obj['word'])+1) # 0 is not masked
    else:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word'])) # 0 is not masked

    

    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])
    
    # clean 0 length data
    if input_sample['seq_len'] == 0:
        print('discard length zero instance: ', target_string)
        return None

    return input_sample

class EEGWordDataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject = 'ALL', eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], setting = 'unique_sent', max_len = 3):
        super().__init__()
        self.word_eeg = []
        self.english_words = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open('vocab.json', 'r', encoding='utf-8') as f:
            # 加载JSON数据到字典
            vocab_dict = json.load(f)
        if not isinstance(input_dataset_dicts,list):
            input_dataset_dicts = [input_dataset_dicts]
        for input_dataset_dict in input_dataset_dicts:
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print('[INFO]using subjects: ', subjects)
            else:
                subjects = [subject]
            
            for key in subjects:
                sub_word_eeg = []
                sub_english_words = []
                sub_labels = []
                for sen_obj in input_dataset_dict[key]:
                    if sen_obj is None:
                        continue
                    for word_obj in sen_obj['word']:
                        word_eeg_embedding = get_word_embedding_eeg_tensor(word_obj, eeg_type, bands)
                        if word_eeg_embedding is None:
                            continue
                        if word_obj['content'] == 'emp11111ty':
                            word_obj['content'] = 'empty'
                        if word_obj['content'] == 'film.1':
                            word_obj['content'] = 'film.'
                        # word_ids = self.tokenizer(word_obj['content'], add_special_tokens=False, return_tensors="pt")['input_ids'][0]
                        # if word_ids.shape[0] > self.max_len:
                        #     continue
                        if word_obj['content'] in vocab_dict:
                            sub_word_eeg.append(word_eeg_embedding)
                            sub_english_words.append(word_obj['content'])
                            sub_labels.append(vocab_dict[word_obj['content']])
                total_num_word = len(sub_english_words)
                train_divider = int(0.8*total_num_word)
                dev_divider = train_divider + int(0.1*total_num_word)
                if phase == 'train':
                    self.word_eeg.append(sub_word_eeg[:train_divider])
                    self.english_words.append(sub_english_words[:train_divider])
                    self.labels.append(sub_labels[:train_divider])
                elif phase == 'dev':
                    self.word_eeg.append(sub_word_eeg[train_divider:dev_divider])
                    self.english_words.append(sub_english_words[train_divider:dev_divider])
                    self.labels.append(sub_labels[train_divider:dev_divider])
                elif phase == 'test':
                    self.word_eeg.append(sub_word_eeg[dev_divider:])
                    self.english_words.append(sub_english_words[dev_divider:])
                    self.labels.append(sub_labels[dev_divider:])
        self.word_eeg = np.concatenate(self.word_eeg)
        self.english_words = list(np.concatenate(self.english_words))
        self.labels = list(np.concatenate(self.labels))
    
    def __len__(self):
        return len(self.english_words)

    def __getitem__(self, idx):
        # word = self.english_words[idx]
        # word_tokenized = self.tokenizer(word, padding='max_length',max_length=self.max_len, add_special_tokens=False, return_tensors="pt")
        # word_id = word_tokenized['input_ids'][0]
        # attention_mask = word_tokenized['attention_mask'][0]
        # return self.word_eeg[idx], word_id, attention_mask
        return self.word_eeg[idx], self.labels[idx]

class EEGSentDataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject = 'ALL', eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], setting = 'unique_sent', max_len = 14, max_sent_len = 56):
        super().__init__()
        self.groud_truth = []
        self.sent_eegs = []
        self.target_ids = []
        self.target_words = []
        self.attention_masks = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_sent_len = max_sent_len

        if not isinstance(input_dataset_dicts,list):
            input_dataset_dicts = [input_dataset_dicts]
        for input_dataset_dict in input_dataset_dicts:
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print('[INFO]using subjects: ', subjects)
            else:
                subjects = [subject]
            
            total_num_sentence = len(input_dataset_dict[subjects[0]])
            
            train_divider = int(0.8*total_num_sentence)
            dev_divider = train_divider + int(0.1*total_num_sentence)
            if phase == 'test':
                for key in subjects:
                    for i in range(dev_divider, total_num_sentence):
                        sen_obj = input_dataset_dict[key][i]
                        if sen_obj is None or len(sen_obj['word']) == 0:
                            continue
                        self.groud_truth.append(sen_obj['content'])
                        sent_eeg = []
                        target_id = []
                        target_word = []
                        attention_mask = []
                        for word_obj in sen_obj['word']:
                            word_eeg_embedding = get_word_embedding_eeg_tensor(word_obj, eeg_type, bands)
                            if word_eeg_embedding is not None:
                                sent_eeg.append(word_eeg_embedding)
                                word = word_obj['content']
                                if word == 'emp11111ty':
                                    word = 'empty'
                                if word == 'film.1':
                                    word = 'film.'
                                word_tokenized = self.tokenizer(word, padding='max_length',max_length=self.max_len, add_special_tokens=False, return_tensors="pt")
                                target_id.append(word_tokenized['input_ids'][0])
                                target_word.append(word)
                                attention_mask.append(word_tokenized['attention_mask'][0])
                        while len(sent_eeg) < self.max_sent_len:
                            sent_eeg.append(torch.zeros(105*len(bands)))
                            target_id.append(torch.zeros(self.max_len))
                            target_word.append('#')
                            attention_mask.append(torch.zeros(self.max_len))
                        self.sent_eegs.append(sent_eeg)
                        self.target_ids.append(target_id)
                        self.target_words.append(target_word)
                        self.attention_masks.append(attention_mask)
    
    def __len__(self):
        return len(self.sent_eegs)

    def __getitem__(self, idx):
        return self.groud_truth[idx], self.sent_eegs[idx], self.target_ids[idx], self.target_words[idx], self.attention_masks[idx]

class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject = 'ALL', eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], setting = 'unique_sent', is_add_CLS_token = False):
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts,list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')
        for input_dataset_dict in input_dataset_dicts:
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print('[INFO]using subjects: ', subjects)
            else:
                subjects = [subject]
            
            total_num_sentence = len(input_dataset_dict[subjects[0]])
            
            train_divider = int(0.8*total_num_sentence)
            dev_divider = train_divider + int(0.1*total_num_sentence)
            
            print(f'train divider = {train_divider}')
            print(f'dev divider = {dev_divider}')

            if setting == 'unique_sent':
                # take first 80% as trainset, 10% as dev and 10% as test
                if phase == 'train':
                    print('[INFO]initializing a train set...')
                    for key in subjects:
                        for i in range(train_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'dev':
                    print('[INFO]initializing a dev set...')
                    for key in subjects:
                        for i in range(train_divider,dev_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'test':
                    print('[INFO]initializing a test set...')
                    for key in subjects:
                        for i in range(dev_divider,total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'pretrain':
                    print('[INFO]initializing a pretrain set...')
                    for key in subjects:
                        for i in range(total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            elif setting == 'unique_subj':
                print('WARNING!!! only implemented for SR v1 dataset ')
                # subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW'] for train
                # subject ['ZMG'] for dev
                # subject ['ZPH'] for test
                if phase == 'train':
                    print(f'[INFO]initializing a train set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH','ZKW']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == 'dev':
                    print(f'[INFO]initializing a dev set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZMG']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == 'test':
                    print(f'[INFO]initializing a test set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZPH']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            print('++ adding task to dataset, now we have:', len(self.inputs))

        print('[INFO]input tensor size:', self.inputs[0]['input_embeddings'].size())
        print()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            # input_sample['rawData'],
            # input_sample['maskedData'],
            input_sample['input_embeddings'], 
            input_sample['seq_len'],
            input_sample['input_attn_mask'], 
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'], 
            input_sample['target_mask'], 
            input_sample['sentiment_label'],  
            input_sample['sent_level_EEG']
        )
        # keys: input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask, 


"""for train classifier on stanford sentiment treebank text-sentiment pairs"""
# class SST_tenary_dataset(Dataset):
#     def __init__(self, ternary_labels_dict, tokenizer, max_len = 56, balance_class = True):
#         self.inputs = []
        
#         pos_samples = []
#         neg_samples = []
#         neu_samples = []

#         for key,value in ternary_labels_dict.items():
#             tokenized_inputs = tokenizer(key, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask = True)
#             input_ids = tokenized_inputs['input_ids'][0]
#             attn_masks = tokenized_inputs['attention_mask'][0]
#             label = torch.tensor(value)
#             # count:
#             if value == 0:
#                 neg_samples.append((input_ids,attn_masks,label))
#             elif value == 1:
#                 neu_samples.append((input_ids,attn_masks,label))
#             elif value == 2:
#                 pos_samples.append((input_ids,attn_masks,label))
#         print(f'Original distribution:\n\tVery positive: {len(pos_samples)}\n\tNeutral: {len(neu_samples)}\n\tVery negative: {len(neg_samples)}')    
#         if balance_class:
#             print(f'balance class to {min([len(pos_samples),len(neg_samples),len(neu_samples)])} each...')
#             for i in range(min([len(pos_samples),len(neg_samples),len(neu_samples)])):
#                 self.inputs.append(pos_samples[i])
#                 self.inputs.append(neg_samples[i])
#                 self.inputs.append(neu_samples[i])
#         else:
#             self.inputs = pos_samples + neg_samples + neu_samples
        
#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, idx):
#         input_sample = self.inputs[idx]
#         return input_sample
#         # keys: input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask, 
        


'''sanity test'''
if __name__ == '__main__':

    check_dataset = 'ZuCo'

    if check_dataset == 'ZuCo':
        whole_dataset_dicts = []
        
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle' 
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        dataset_path_task2_v2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle' 
        with open(dataset_path_task2_v2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        print()
        for key in whole_dataset_dicts[0]:
            print(f'task2_v2, sentence num in {key}:',len(whole_dataset_dicts[0][key])) 
        print()

        tokenizer = BartTokenizer.from_pretrained('bart-large')
        dataset_setting = 'unique_sent'
        subject_choice = 'ALL'
        print(f'![Debug]using {subject_choice}')
        eeg_type_choice = 'GD'
        print(f'[INFO]eeg type {eeg_type_choice}') 
        bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
        print(f'[INFO]using bands {bands_choice}')
        # train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
        # train_dataloader = DataLoader(train_set, batch_size = 8, shuffle=True, num_workers=4)
        # dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
        # test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

        # train_set = EEGWordDataset(whole_dataset_dicts, 'train', tokenizer, subject_choice, eeg_type_choice, bands_choice, dataset_setting)
        # dev_set = EEGWordDataset(whole_dataset_dicts, 'dev', tokenizer, subject_choice, eeg_type_choice, bands_choice, dataset_setting)
        test_set = EEGWordDataset(whole_dataset_dicts, 'test', tokenizer, subject_choice, eeg_type_choice, bands_choice, dataset_setting)
        # test_sent_set = EEGSentDataset(whole_dataset_dicts, 'test', tokenizer, subject_choice, eeg_type_choice, bands_choice, dataset_setting)

        # print('trainset size:',len(train_set))
        # print('devset size:',len(dev_set))
        # print('testset size:',len(test_set))

    # elif check_dataset == 'stanford_sentiment':
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    #     SST_dataset = SST_tenary_dataset(SST_SENTIMENT_LABELS, tokenizer)
    #     print('SST dataset size:',len(SST_dataset))
    #     print(SST_dataset[0])
    #     print(SST_dataset[1])
