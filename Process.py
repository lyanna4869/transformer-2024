# # import pandas as pd
# # import torchtext
# # from torchtext.legacy import data
# # from Tokenize import tokenize
# # from Batch import MyIterator, batch_size_fn
# # import os
# # import dill as pickle

# # def read_data(opt):
    
# #     if opt.src_data is not None:
# #         try:
# #             opt.src_data = open(opt.src_data).read().strip().split('\n')
# #         except:
# #             print("error: '" + opt.src_data + "' file not found")
# #             quit()
    
# #     if opt.trg_data is not None:
# #         try:
# #             opt.trg_data = open(opt.trg_data).read().strip().split('\n')
# #         except:
# #             print("error: '" + opt.trg_data + "' file not found")
# #             quit()

# # def create_fields(opt):
    
# #     spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it', 'nl']
# #     if opt.src_lang not in spacy_langs:
# #         print('invalid src language: ' + opt.src_lang + 'supported languages : ' + str(spacy_langs))
# #     if opt.trg_lang not in spacy_langs:
# #         print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + str(spacy_langs))
    
# #     print("loading spacy tokenizers...")
    
# #     t_src = tokenize(opt.src_lang)
# #     t_trg = tokenize(opt.trg_lang)

# #     TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
# #     SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

# #     if opt.load_weights is not None:
# #         try:
# #             print("loading presaved fields...")
# #             SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
# #             TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
# #         except:
# #             print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
# #             quit()
        
# #     return(SRC, TRG)

# # def create_dataset(opt, SRC, TRG):

# #     print("creating dataset and iterator... ")


# #     raw_data = {'src' : [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
# #     df = pd.DataFrame(raw_data, columns=["src", "trg"], dtype=str)
# #     mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
# #     df = df.loc[mask]

# #     df.to_csv("translate_transformer_temp.csv", index=False)
    
# #     data_fields = [('src', SRC), ('trg', TRG)]
# #     train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

# #     train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
# #                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
# #                         batch_size_fn=batch_size_fn, train=True, shuffle=True)
    
# #     os.remove('translate_transformer_temp.csv')

# #     if opt.load_weights is None:
# #         SRC.build_vocab(train)
# #         TRG.build_vocab(train)
# #         if opt.checkpoint > 0:
# #             try:
# #                 os.mkdir("weights")
# #             except:
# #                 print("weights folder already exists, run program with -load_weights weights to load them")
# #                 quit()
# #             pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
# #             pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

# #     opt.src_pad = SRC.vocab.stoi['<pad>']
# #     opt.trg_pad = TRG.vocab.stoi['<pad>']

# #     opt.train_len = get_len(train_iter)

# #     return train_iter

# # def get_len(train):

# #     for i, b in enumerate(train):
# #         pass
    
# #     return i

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from Tokenize import tokenize
import os
import dill as pickle

class TranslationDataset(Dataset):
    def __init__(self, src_data, trg_data, src_tokenizer, trg_tokenizer, max_strlen):
        self.src_data = src_data
        self.trg_data = trg_data
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.max_strlen = max_strlen

        self.filtered_data = [
            (src, trg)
            for src, trg in zip(self.src_data, self.trg_data)
            if len(self.src_tokenizer(src)) < self.max_strlen and len(self.trg_tokenizer(trg)) < self.max_strlen
        ]

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        src, trg = self.filtered_data[idx]
        return src, trg

def read_data(opt):
    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data).read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()

    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data).read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()

def create_fields(opt):
    spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it', 'nl']
    if opt.src_lang not in spacy_langs:
        print(f"invalid src language: {opt.src_lang}. Supported languages: {spacy_langs}")
    if opt.trg_lang not in spacy_langs:
        print(f"invalid trg language: {opt.trg_lang}. Supported languages: {spacy_langs}")

    print("loading spacy tokenizers...")

    src_tokenizer = get_tokenizer("spacy", language=opt.src_lang) # 获取一个分词器(spacy)
    trg_tokenizer = get_tokenizer("spacy", language=opt.trg_lang)

    return src_tokenizer, trg_tokenizer

def build_vocab(dataset, tokenizer):
    def yield_tokens(data_iter):
        for src, trg in data_iter:
            yield tokenizer(src)
            yield tokenizer(trg)

    return build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>", "<pad>", "<sos>", "<eos>"], special_first=True)

def create_dataset(opt, src_tokenizer, trg_tokenizer):
    print("creating dataset and iterator... ")

    dataset = TranslationDataset(
        src_data=opt.src_data, # 存放数据的路径
        trg_data=opt.trg_data,
        src_tokenizer=src_tokenizer, # 分词器
        trg_tokenizer=trg_tokenizer,
        max_strlen=opt.max_strlen
    )

    src_vocab = build_vocab(dataset, src_tokenizer) # 构建词汇表
    trg_vocab = build_vocab(dataset, trg_tokenizer)

    src_vocab.set_default_index(src_vocab["<unk>"]) #为 词汇表（Vocabulary） 设置默认索引（Default Index），
    # 用于处理在翻译任务或其他 NLP 任务中出现的 未知词（unknown tokens, <unk>）
    trg_vocab.set_default_index(trg_vocab["<unk>"])

    dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, collate_fn=lambda x: collate_fn(x, src_vocab, trg_vocab))

    if opt.checkpoint > 0:
        os.makedirs("weights", exist_ok=True)
        pickle.dump(src_vocab, open('weights/SRC.pkl', 'wb'))
        pickle.dump(trg_vocab, open('weights/TRG.pkl', 'wb'))

    opt.src_pad = src_vocab["<pad>"]
    opt.trg_pad = trg_vocab["<pad>"]

    opt.train_len = len(dataset)

    return dataloader, src_vocab, trg_vocab

from collections import namedtuple

Batch = namedtuple("Batch", ["src", "trg"])

def collate_fn(batch, src_vocab, trg_vocab):
    src_batch, trg_batch = zip(*batch)
    src_batch = [torch.tensor([src_vocab[token] for token in src.split()]) for src in src_batch]
    trg_batch = [torch.tensor([trg_vocab[token] for token in trg.split()]) for trg in trg_batch]

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=src_vocab["<pad>"])
    trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, padding_value=trg_vocab["<pad>"])

    return Batch(src=src_batch, trg=trg_batch)



def get_len(dataset):
    return len(dataset)
