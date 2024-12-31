# import torch
# import numpy as np
# from torch.autograd import Variable


# def nopeak_mask(size, opt):
#     np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
#     np_mask = Variable(torch.from_numpy(np_mask == 0).to(opt.device))
#     return np_mask

# def create_masks(src, trg, opt):
    
#     src_mask = (src != opt.src_pad).unsqueeze(-2).to(opt.device)

#     if trg is not None:
#         trg_mask = (trg != opt.trg_pad).unsqueeze(-2).to(opt.device)
#         size = trg.size(1)  # get seq_len for matrix
#         np_mask = nopeak_mask(size, opt).to(opt.device)
#         trg_mask = trg_mask & np_mask

#     else:
#         trg_mask = None
#     return src_mask, trg_mask

# # patch on Torchtext's batching process that makes it more efficient
# # from http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks

# class MyIterator(data.Iterator):
#     def create_batches(self):
#         if self.train:
#             def pool(d, random_shuffler):
#                 for p in data.batch(d, self.batch_size * 100):
#                     p_batch = data.batch(
#                         sorted(p, key=self.sort_key),
#                         self.batch_size, self.batch_size_fn)
#                     for b in random_shuffler(list(p_batch)):
#                         yield b
#             self.batches = pool(self.data(), self.random_shuffler)
            
#         else:
#             self.batches = []
#             for b in data.batch(self.data(), self.batch_size,
#                                           self.batch_size_fn):
#                 self.batches.append(sorted(b, key=self.sort_key))

# global max_src_in_batch, max_tgt_in_batch

# def batch_size_fn(new, count, sofar):
#     "Keep augmenting batch and calculate total number of tokens + padding."
#     global max_src_in_batch, max_tgt_in_batch
#     if count == 1:
#         max_src_in_batch = 0
#         max_tgt_in_batch = 0
#     max_src_in_batch = max(max_src_in_batch,  len(new.src))
#     max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
#     src_elements = count * max_src_in_batch
#     tgt_elements = count * max_tgt_in_batch
#     return max(src_elements, tgt_elements)


import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable

def nopeak_mask(size, device):
    """
    Generate a mask to prevent attention to future tokens (causal masking).
    """
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask == 0).to(device))
    return np_mask

def create_masks(src, trg, src_pad, trg_pad, device):
    """
    Create source and target masks.
    """
    src_mask = (src != src_pad).unsqueeze(-2).to(device)

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2).to(device)
        size = trg.size(1)  # Sequence length
        np_mask = nopeak_mask(size, device).to(device)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None

    return src_mask, trg_mask

class TranslationDataset(Dataset):
    """
    Custom Dataset for source and target sequences.
    """
    def __init__(self, src_data, trg_data, src_vocab, trg_vocab):
        self.src_data = src_data
        self.trg_data = trg_data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src = [self.src_vocab[token] for token in self.src_data[idx].split()]
        trg = [self.trg_vocab[token] for token in self.trg_data[idx].split()]
        return torch.tensor(src), torch.tensor(trg)

def batch_size_fn(new, count, sofar):
    """
    Calculate the number of tokens in the batch for efficient batching.
    """
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new[0]))  # new[0] is src
    max_tgt_in_batch = max(max_tgt_in_batch, len(new[1]) + 2)  # new[1] is trg
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

def collate_fn(batch, src_pad, trg_pad):
    """
    Custom collate function for padding and batching.
    """
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=src_pad)
    trg_batch = pad_sequence(trg_batch, padding_value=trg_pad)
    return src_batch, trg_batch

def create_dataloader(src_data, trg_data, src_vocab, trg_vocab, batch_size, src_pad, trg_pad, shuffle=True):
    """
    Create DataLoader for batching.
    """
    dataset = TranslationDataset(src_data, trg_data, src_vocab, trg_vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=lambda x: collate_fn(x, src_pad, trg_pad))
    return dataloader
