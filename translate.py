import argparse
import time
import torch
import numpy as np
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import pdb
import dill as pickle
import argparse
from Models import get_model
from Beam import beam_search
from nltk.corpus import wordnet
from torch.autograd import Variable
import re

def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]
            
    return 0

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

# def preprocess_sentence(sentence, vocab, tokenizer):
#     tokens = tokenizer(sentence)  # 使用分词器将句子分词
#     indexed = [vocab[token] for token in tokens]  # 将词转换为索引
#     return torch.tensor(indexed, dtype=torch.long)

# def translate_sentence(sentence, model, opt, SRC, TRG,src_tokenizer, trg_vocab):
    
#     model.eval()
#     indexed = preprocess_sentence(sentence, SRC, src_tokenizer)
#     src_tensor = indexed.unsqueeze(0).to(opt.device)  # 添加 batch 维度

#     src_mask = (src_tensor != SRC["<pad>"]).unsqueeze(-2).to(opt.device)
#     trg_init_token = TRG["<sos>"]
#     trg_tensor = torch.tensor([trg_init_token], dtype=torch.long, device=opt.device).unsqueeze(0)
#     # indexed = []
#     # sentence = SRC.preprocess(sentence)
#     for tok in sentence:
#         if SRC.vocab.stoi[tok] != 0 or opt.floyd is True:
#             indexed.append(SRC.vocab.stoi[tok])
#         else:
#             indexed.append(get_synonym(tok, SRC))
#     sentence = Variable(torch.LongTensor([indexed]), device=opt.device)

#     sentence = beam_search(sentence, model, SRC, TRG, opt)

#     return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)

def preprocess_sentence(sentence, vocab, tokenizer):
    """
    对输入句子进行分词并转化为索引。
    """
    tokens = tokenizer(sentence)  # 使用分词器将句子分词
    indexed = [vocab[token] for token in tokens]  # 将词转换为索引
    return torch.tensor(indexed, dtype=torch.long)

def nopeak_mask(size, device):
    """
    创建一个上三角掩码，用于目标序列的自注意力机制，防止预测未来的词。
    """
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = torch.from_numpy(np_mask) == 0
    return np_mask.to(device)

def translate_sentence(sentence, model, opt, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer):
    """
    翻译单个句子。
    """
    model.eval()
    # 将输入句子预处理为索引
    src_tensor = preprocess_sentence(sentence, src_vocab, src_tokenizer).unsqueeze(0).to(opt.device)

    # 创建 src_mask
    src_mask = (src_tensor != src_vocab["<pad>"]).unsqueeze(-2).to(opt.device)

    # 初始化目标序列（以 <sos> 开始）
    trg_tensor = torch.tensor([trg_vocab["<sos>"]], dtype=torch.long, device=opt.device).unsqueeze(0)

    # 逐步生成翻译
    for _ in range(opt.max_len):
        trg_mask = nopeak_mask(trg_tensor.size(1), opt.device)
        preds = model(src_tensor, trg_tensor, src_mask, trg_mask)
        next_word = preds.argmax(2)[:, -1].item()  # 获取概率最高的词
        trg_tensor = torch.cat((trg_tensor, torch.tensor([[next_word]], device=opt.device)), dim=1)

        # 如果生成了 <eos>，则停止生成
        if next_word == trg_vocab["<eos>"]:
            break

    # 将生成的索引转回单词
    translated_tokens = [trg_vocab.lookup_token(idx) for idx in trg_tensor[0].tolist()]

    # 返回翻译的句子，去掉 <sos> 和 <eos>
    return " ".join(translated_tokens[1:-1])

def translate(opt, model, SRC, TRG):
    sentences = opt.text.lower().split('.')
    translated = []

    for sentence in sentences:
        translated.append(translate_sentence(sentence + '.', model, opt, SRC, TRG).capitalize())

    return (' '.join(translated))


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', default='./data/english.txt')
    parser.add_argument('-trg_data', default='./data/french.txt')
    parser.add_argument('-src_lang', default= 'en_core_web_sm')
    parser.add_argument('-trg_lang', default= 'fr_core_news_sm')
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    # parser.add_argument('-src_lang', required=True)
    # parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-batchsize', type=int, default=128)
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    
    opt = parser.parse_args()

    opt.device = 'cuda' if opt.no_cuda is False else 'cpu'
 
    assert opt.k > 0
    assert opt.max_len > 10

    SRC, TRG = create_fields(opt)
    opt.train, src_vocab, trg_vocab = create_dataset(opt, SRC, TRG)
    model = get_model(opt, len(src_vocab), len(trg_vocab))

    
    while True:
        opt.text =input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
        if opt.text=="q":
            break
        if opt.text=='f':
            fpath =input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
            try:
                opt.text = ' '.join(open(opt.text, encoding='utf-8').read().split('\n'))
            except:
                print("error opening or reading text file")
                continue
        phrase = translate(opt, model, SRC, TRG)
        print('> '+ phrase + '\n')

if __name__ == '__main__':
    main()
