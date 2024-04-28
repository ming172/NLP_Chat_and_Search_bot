import random
import jieba
import torch
import logging
from train_model_net import EncoderRNN, LuongAttnDecoderRNN
from data_load import get_dataloader
from config import Config

jieba.setLogLevel(logging.INFO)


# 计算带掩码的负对数似然损失
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()  # 总数
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))  # 交叉熵
    loss = crossEntropy.masked_select(mask.bool()).mean()  # 选择掩码中为True的元素求平均
    return loss, nTotal.item()


# 批量训练
def train_by_batch(sos, opt, data, encoder_optimizer, decoder_optimizer, encoder, decoder):
    encoder_optimizer.zero_grad()  # 编码器优化器梯度清零
    decoder_optimizer.zero_grad()  # 解码器优化器梯度清零

    inputs, targets, mask, input_lengths, max_target_length, indexes = data  # 输入、目标、掩码、输入长度、目标序列最大长度、索引
    inputs = inputs.to(opt.device)
    targets = targets.to(opt.device)
    mask = mask.to(opt.device)
    input_lengths = input_lengths.to(opt.device)

    loss = 0
    print_losses = []  # 记录每个batch的损失
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(inputs, input_lengths)  # 编码器前向传播
    decoder_input = torch.LongTensor([[sos for _ in range(opt.batch_size)]])  # 解码器输入
    decoder_input = decoder_input.to(opt.device)
    decoder_hidden = encoder_hidden[:decoder.num_layers]  # 解码器隐藏状态

    use_teacher_forcing = True if random.random() < opt.teacher_forcing_ratio else False  # 是否使用Teacher Forcing

    if use_teacher_forcing:
        # 使用Teacher Forcing
        for t in range(max_target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_input = targets[t].view(1, -1)  # 下一个解码器输入为当前目标序列的值

            mask_loss, nTotal = maskNLLLoss(decoder_output, targets[t], mask[t])  # 计算损失
            mask_loss = mask_loss.to(opt.device)
            loss += mask_loss

            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        # 不使用Teacher Forcing
        for t in range(max_target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(opt.batch_size)]])
            decoder_input = decoder_input.to(opt.device)

            mask_loss, nTotal = maskNLLLoss(decoder_output, targets[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # 反向传播
    loss.backward()

    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), opt.clip)  # 梯度裁剪
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), opt.clip)

    encoder_optimizer.step()  # 更新编码器参数
    decoder_optimizer.step()  # 更新解码器参数

    return sum(print_losses) / n_totals  # 平均损失


# 训练函数
def train(**kwargs):
    opt = Config()  # 获取配置
    for k, v in kwargs.items():
        setattr(opt, k, v)

    dataloader = get_dataloader(opt)  # 获取数据加载器
    _data = dataloader.dataset._data  # 获取数据
    word2ix = _data['word2ix']  # 获取词到索引的映射
    sos = word2ix.get(_data.get('sos'))  # SOS的索引
    voc_length = len(word2ix)  # 词汇表大小

    encoder = EncoderRNN(opt, voc_length)  # 创建编码器
    decoder = LuongAttnDecoderRNN(opt, voc_length)  # 创建解码器

    if opt.model_ckpt_yn:
        checkpoint = torch.load(opt.model_ckpt)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])

    encoder = encoder.to(opt.device)
    decoder = decoder.to(opt.device)
    encoder.train()
    decoder.train()

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=opt.learning_rate * opt.decoder_learning_ratio)
    if opt.model_ckpt_yn:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])

    print_loss = 0

    for epoch in range(opt.epoch):
        for ii, data in enumerate(dataloader):
            loss = train_by_batch(sos, opt, data, encoder_optimizer, decoder_optimizer, encoder, decoder)
            print_loss += loss
            if ii % opt.print_every == 0:
                print_loss_avg = print_loss / opt.print_every
                print("Epoch: {}; Epoch Percent complete: {:.1f}%; Average loss: {:.4f}".format(epoch,
                                                                                                epoch / opt.epoch * 100,
                                                                                                print_loss_avg))
                print_loss = 0

        if epoch % opt.save_every == 0 and epoch != 0:
            save_path = '{prefix}_{epoch}_{print_loss_avg:.2f}.pth'.format(prefix=opt.prefix, epoch=epoch,
                                                                           print_loss_avg=print_loss_avg)
            torch.save({
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
            }, save_path)


if __name__ == "__main__":
    train()
