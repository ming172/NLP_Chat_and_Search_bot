

import jieba
import torch
import re
import logging
jieba.setLogLevel(logging.INFO)

corpus_file = 'new_tsv/zhdd_lines -min_new.tsv' #待处理的对话数据集
cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]") #分词处理正则
unknown = '</UNK>' #unknown字符
eos = '</EOS>' #句子结束符
sos = '</SOS>' #句子开始符
padding = '</PAD>' #句子填充负
max_voc_length = 10000 #字典最大长度
min_word_appear = 1 #加入字典的词的词频最小值
save_path = 'new_tsv/processed_conversation_dataset.pth' #将要处理完的对话数据集保存路径

def preprocess():
    print("开始处理对话数据集，第一步分词进度如下：")
    data = []
    with open(corpus_file, encoding='GBK') as f:
        lines = f.readlines()
        i=0;j=0
        for line in lines:
            i+=1
            values = line.strip('\n').split('\t')
            sentences = []
            for v in range(0,len(values)):
                value=values[v]
                # print("value:",value,values)
                # value: 昆明 那里 配 眼镜 比较 便宜
                sentence = jieba.lcut(cop.sub("",value))#未分词用这个
                # sentence = value.split(' ')#已分词用这个
                # print("sentence1 分词结果:", sentence)
                #sentence1 分词结果: ['昆明', '那里', '配眼镜', '比较', '便宜']
                sentence = sentence + [eos]
                # print("sentence2:", sentence)
                sentences.append(sentence)
                # print("sentences3:",sentences)
            data.append(sentences)
            per_1=i/len(lines)
            if per_1 - j >= 0.01:
                j = per_1
                print(f'已完成{per_1 * 100:.2f}%')

    print("第二步生成字典和句子索引：")
    word_nums = {}
    def update(word_nums):
        def fun(word):
            word_nums[word] = word_nums.get(word, 0) + 1
            return None
        return fun
    lambda_ = update(word_nums)
    _ = {lambda_(word) for sentences in data for sentence in sentences for word in sentence}

    word_nums_list = sorted([(num, word) for word, num in word_nums.items()], reverse=True)

    words = [word[1] for word in word_nums_list[:max_voc_length] if word[0] >= min_word_appear]

    words = [unknown, padding, sos] + words
    word2ix = {word: ix for ix, word in enumerate(words)}
    ix2word = {ix: word for word, ix in word2ix.items()}
    ix_corpus = [[[word2ix.get(word, word2ix.get(unknown)) for word in sentence]
                        for sentence in item]
                        for item in data]

    clean_data = {
        'corpus': ix_corpus,
        'word2ix': word2ix,
        'ix2word': ix2word,
        'unknown' : '</UNK>',
        'eos' : '</EOS>',
        'sos' : '</SOS>',
        'padding': '</PAD>',
    }
    torch.save(clean_data, save_path)
    print('save clean data in %s' % save_path)

if __name__ == "__main__":
    preprocess()