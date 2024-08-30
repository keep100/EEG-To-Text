import pickle
import re
import matplotlib.pyplot as plt
import json
from transformers import BartTokenizer

if __name__ == '__main__':
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

    def get_vocab(sent):
        sent_clean = re.sub(r'[^\w\s-]', ' ', sent)
        sent_clean = re.sub(r'\d', ' ', sent_clean)
        return sent_clean.split()
    #ZAB ZAB YRH
    vocab_set = set()
    for item in whole_dataset_dicts[0]['ZAB']:
        if item is None:
            continue
        vocab_list = get_vocab(item['content'])
        for word in vocab_list:
            vocab_set.add(word)
    for item in whole_dataset_dicts[1]['ZAB']:
        if item is None:
            continue
        vocab_list = get_vocab(item['content'])
        for word in vocab_list:
            vocab_set.add(word)
    for item in whole_dataset_dicts[2]['YRH']:
        if item is None:
            continue
        vocab_list = get_vocab(item['content'])
        for word in vocab_list:
            vocab_set.add(word)
    print(len(vocab_set))
    
    tokenizer_dict = {}
    vocab_dict = {}
    tokenizer = BartTokenizer.from_pretrained("bart-large")
    for word in vocab_set:
        ids = tokenizer(word,add_special_tokens=False,return_tensors='pt')['input_ids'][0]
        ids_len = str(len(ids))
        if len(ids) <= 3:
            vocab_dict[word] = len(vocab_dict)
        if ids_len in tokenizer_dict:
            tokenizer_dict[ids_len] = tokenizer_dict[ids_len]+1
        else:
            tokenizer_dict[ids_len] = 1
    print(tokenizer_dict)
    # with open('vocab.json', 'w', encoding='utf-8') as f:
    #     json.dump(vocab_dict, f, ensure_ascii=False)
    # keys = sorted(tokenizer_dict.keys(),key=lambda id_len: int(id_len))
    # values = [tokenizer_dict[key] for key in keys]

    # # 绘制柱状图
    # plt.figure(figsize=(10, 6))  # 设置图形的大小
    # plt.bar(keys, values, color='skyblue')  # 创建柱状图

    # # 添加标题和标签
    # plt.title('Bar Chart of Dictionary Data')
    # plt.xlabel('Keys')
    # plt.ylabel('Values')

    # plt.savefig('bar_chart.png', format='png')