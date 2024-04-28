import re
import jieba
import torch
import logging
from train_model_net import EncoderRNN, LuongAttnDecoderRNN
from greedy_search import GreedySearchDecoder
from data_load import get_dataloader
jieba.setLogLevel(logging.INFO)



def generate(input_seq, searcher, sos, eos, opt):
    input_batch = [input_seq]
    input_lengths = torch.tensor([len(seq) for seq in input_batch])
    input_batch = torch.LongTensor([input_seq]).transpose(0,1)
    input_batch = input_batch.to(opt.device)
    input_lengths = input_lengths.to(opt.device)
    tokens, scores = searcher(sos, eos, input_batch, input_lengths, opt.max_generate_length, opt.device)
    return tokens

def test(opt):

    dataloader = get_dataloader(opt)
    _data = dataloader.dataset._data
    word2ix,ix2word = _data['word2ix'], _data['ix2word']
    sos = word2ix.get(_data.get('sos'))
    eos = word2ix.get(_data.get('eos'))
    unknown = word2ix.get(_data.get('unknown'))
    voc_length = len(word2ix)

    encoder = EncoderRNN(opt, voc_length)
    decoder = LuongAttnDecoderRNN(opt, voc_length)

    if opt.model_ckpt == None:
        raise ValueError('model_ckpt is None.')
        return False
    checkpoint = torch.load(opt.model_ckpt, map_location=lambda s, l: s)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    with torch.no_grad():
        encoder = encoder.to(opt.device)
        decoder = decoder.to(opt.device)
        encoder.eval()
        decoder.eval()
        searcher = GreedySearchDecoder(encoder, decoder)
        return searcher, sos, eos, unknown, word2ix, ix2word

def output_answer(input_sentence, searcher, sos, eos, unknown, opt, word2ix, ix2word):
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
    input_seq = jieba.lcut(cop.sub("",input_sentence))
    input_seq = input_seq[:opt.max_input_length] + ['</EOS>']
    input_seq = [word2ix.get(word, unknown) for word in input_seq]
    tokens = generate(input_seq, searcher, sos, eos, opt)
    output_words = ''.join([ix2word[token.item()] for token in tokens if token.item() != eos])
    return output_words