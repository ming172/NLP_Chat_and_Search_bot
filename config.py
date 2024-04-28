import torch

class Config:
    # 配置文件路径
    corpus_data_path = 'new_tsv/processed_conversation_dataset.pth'
    mysql_QA = True  # 是否载入计算机知识库
    # 输入和生成的最大句子长度
    max_input_length = 50
    max_generate_length = 20
    # 模型相关
    prefix = 'chatbot_model'  # 模型前缀名称
    model_ckpt = 'chatbot_model.pth'  # 加载的模型路径
    model_ckpt_yn = False  # 是否需要断点加载，用之前的模型继续训练
    # 训练超参数
    batch_size = 1024  # 提示GPU内存不够时1024，够则可尝试调大
    shuffle = True  # 是否打乱数据
    num_workers = 0  # 数据加载器的多进程数
    bidirectional = True  # Encoder-RNN是否双向
    hidden_size = 256
    embedding_dim = 256
    method = 'dot'  # 注意力机制计算方法
    dropout = 0  # 是否使用dropout
    clip = 50.0  # 梯度裁剪阈值
    num_layers = 2  # Encoder-RNN层数
    learning_rate = 1e-4 * 2  # 学习率根据本人的尝试得出，可根据训练语料的不同，视初步训练结果自行调节
    teacher_forcing_ratio = 1.0  # Teacher Forcing比例
    decoder_learning_ratio = 5.0
    # 训练过程参数
    epoch = 6000  # 总共训练次数，其实不需要训练这么多次，根据收敛的需要即使中断程序
    print_every = 1  # 每隔几次epoch打印一次
    save_every = 10  # 每隔几个Epoch保存一次模型，= 10存储及时不怕中断

    # 设备设置
    use_gpu = torch.cuda.is_available()  # 是否可用GPU
    device = torch.device("cuda" if use_gpu else "cpu")
