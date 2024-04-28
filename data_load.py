import torch
import itertools
from torch.utils import data as dataimport

def zeroPadding(l, fillvalue):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def create_collate_fn(padding, eos):
    def collate_fn(corpus_item):
        corpus_item.sort(key=lambda p: len(p[0]), reverse=True) 
        inputs, targets, indexes = zip(*corpus_item)
        input_lengths = torch.tensor([len(inputVar) for inputVar in inputs])
        inputs = zeroPadding(inputs, padding)
        inputs = torch.LongTensor(inputs)
        
        max_target_length = max([len(targetVar) for targetVar in targets])
        targets = zeroPadding(targets, padding)
        mask = binaryMatrix(targets, padding)
        mask = torch.ByteTensor(mask)
        targets = torch.LongTensor(targets)
        
        return inputs, targets, mask, input_lengths, max_target_length, indexes

    return collate_fn




corpus_index=0#文本处理过程中出了一个小漏洞，源头出在tsv文本上，但文本行数太多，暂时找不到，因为此漏洞影响很小，因此这样处理
class CorpusDataset(dataimport.Dataset):

    def __init__(self, opt):
        self.opt = opt
        self._data = torch.load(opt.corpus_data_path)
        self.word2ix = self._data['word2ix']
        self.corpus = self._data['corpus']
        self.padding = self.word2ix.get(self._data.get('padding'))
        self.eos = self.word2ix.get(self._data.get('eos'))
        self.sos = self.word2ix.get(self._data.get('sos'))

    def __getitem__(self, index):
        inputVar = self.corpus[index][0]
        if len(self.corpus[index])==1:#文本处理过程中出了一个小漏洞，源头出在tsv文本上，但文本行数太多，暂时找不到，因为此漏洞影响很小，因此这样处理
            global corpus_index
            corpus_index+=1
            print(self.corpus[index])
            print(corpus_index)
            self.corpus[index].append(inputVar)
        targetVar = self.corpus[index][1]

        return inputVar, targetVar, index

    def __len__(self):
        return len(self.corpus)


def get_dataloader(opt):
    dataset = CorpusDataset(opt)
    dataloader = dataimport.DataLoader(dataset,
                                       batch_size=opt.batch_size,
                                       shuffle=opt.shuffle,
                                       num_workers=opt.num_workers,
                                       drop_last=True,
                                       collate_fn=create_collate_fn(dataset.padding, dataset.eos))
    return dataloader