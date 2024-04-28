


corpus_file = 'zhdd_lines -min' #原tsv文件名称，未处理的数据

max_sentence_length = 50 #最大句子长度

file_path = f"new_tsv/{corpus_file}_new.tsv"#将一行多轮对话变成一行一轮对话，形成的新tsv文件路径
def append_to_file(file_path, content):
    # 打开文件，如果文件不存在则创建它，以追加模式打开
    with open(file_path, 'a') as file:
        # 在文件末尾写入内容，并添加换行符
        file.write('\n'+content)

#处理对话数据集
def preprocess():
    print("开始将一行多轮对话变成一行一轮对话，并形成的新的tsv文件，进度如下：")
    with open(f'{corpus_file}.tsv', encoding='utf-8') as f:
        lines = f.readlines()
        i=0;j=0
        for line in lines:
            i+=1
            talks= line.strip('\n').split('\t')
            for t in range(0,len(talks)-1):
                talk0=talks[t]
                talk1 = talks[t+1]
                one_talk = talk0[:max_sentence_length]+'\t'+talk1[:max_sentence_length]
                # print(one_talk)
                append_to_file(file_path, one_talk)
            per_1=i/len(lines)
            if per_1-j>=0.01:
                j=per_1
                print(f'已完成{per_1*100:.2f}%')

if __name__ == "__main__":
    preprocess()