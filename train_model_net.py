import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, opt, voc_length):
        super(EncoderRNN, self).__init__()
        self.num_layers = opt.num_layers  # 编码器GRU层数
        self.hidden_size = opt.hidden_size  # 编码器GRU隐藏层尺寸
        self.embedding = nn.Embedding(voc_length, opt.embedding_dim)  # 词嵌入层
        self.gru = nn.GRU(opt.embedding_dim, self.hidden_size,
                          self.num_layers, dropout=(0 if opt.num_layers == 1 else opt.dropout),
                          bidirectional=opt.bidirectional)  # GRU层

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)  # 词嵌入
        input_lengths_cpu = input_lengths.cpu()  # 将长度转移到CPU上
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths_cpu)  # 打包填充序列
        outputs, hidden = self.gru(packed, hidden)  # GRU前向传播
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # 解包填充序列
        # 将双向GRU输出相加以得到最终输出
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class Attn(torch.nn.Module):
    def __init__(self, attn_method, hidden_size):
        super(Attn, self).__init__()
        self.method = attn_method  # 注意力计算方法
        self.hidden_size = hidden_size  # 隐藏层尺寸
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))

    def dot_score(self, hidden, encoder_outputs):
        return torch.sum(hidden * encoder_outputs, dim=2)

    def general_score(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_outputs):
        energy = self.attn(torch.cat((hidden.expand(encoder_outputs.size(0), -1, -1),
                                      encoder_outputs), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()

        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, opt, voc_length):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_method = opt.method  # 注意力计算方法
        self.hidden_size = opt.hidden_size  # 隐藏层尺寸
        self.output_size = voc_length  # 输出词汇表大小
        self.num_layers = opt.num_layers  # 解码器GRU层数
        self.dropout = opt.dropout  # dropout率
        self.embedding = nn.Embedding(voc_length, opt.embedding_dim)  # 词嵌入层
        self.embedding_dropout = nn.Dropout(self.dropout)  # 词嵌入层的dropout
        self.gru = nn.GRU(opt.embedding_dim, self.hidden_size, self.num_layers,
                          dropout=(0 if self.num_layers == 1 else self.dropout))  # GRU层
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)  # 将解码器GRU隐藏状态和注意力权重拼接的线性层
        self.out = nn.Linear(self.hidden_size, self.output_size)  # 输出层
        self.attn = Attn(self.attn_method, self.hidden_size)  # 注意力层

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)  # 词嵌入
        embedded = self.embedding_dropout(embedded)  # 词嵌入dropout
        rnn_output, hidden = self.gru(embedded, last_hidden)  # GRU前向传播
        attn_weights = self.attn(rnn_output, encoder_outputs)  # 计算注意力权重
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # 加权求和得到上下文
        rnn_output = rnn_output.squeeze(0)  # 去除维度为1的维度
        context = context.squeeze(1)  # 去除维度为1的维度
        concat_input = torch.cat((rnn_output, context), 1)  # 拼接解码器GRU隐藏状态和上下文
        concat_output = torch.tanh(self.concat(concat_input))  # 进行线性变换并应用tanh激活函数
        output = self.out(concat_output)  # 输出
        output = F.softmax(output, dim=1)  # softmax得到概率分布

        return output, hidden
