# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
import copy
import torch.nn.functional as F
class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'Dilatedc2'
        self.train_path = dataset + '/data/weibo/train_weibo.txt'  # 训练集
        self.dev_path = dataset + '/data/weibo/valid_weibo.txt'  # 验证集
        self.test_path = dataset + '/data/weibo/test_weibo.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/weibo/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.class_list_fusion = [x.strip() for x in open(
            dataset + '/data/weibo/class3.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.2                                             # 随机失活
        self.require_improvement = 600                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size =20                                           # mini-batch大小
        self.pad_size = 64                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128                                          # lstm隐藏层
        self.hidden_structs = [128] * 4                                 # 扩张层结构
        self.num_layers = 2  # lstm层数
        self.dilations = [1, 2, 4, 8]                                         # 扩张指数
        self.input_dim = 300
        self.cell_type = 'LSTM'
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 64
'''Dilated RNN'''


def _rnn_reformat(x, input_dims, n_steps):
    """
    This function reformat input to the shape that standard RNN can take.

    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
    Outputs:
        x_reformat -- a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
    """
    # permute batch_size and n_steps
    x_ = x.permute(1, 0, 2).contiguous()
    # reshape to (n_steps*batch_size, input_dims)
    x_ = x_.view(-1, input_dims)
    # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
    x_reformat = torch.chunk(x_, n_steps, 0)

    return x_reformat

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.device = config.device
        self.hidden_structs = config.hidden_structs
        self.dilations = config.dilations
        self.input_dim = config.input_dim
        self.n_steps = config.pad_size
        self.cell_type = config.cell_type
        self.batch_size = config.batch_size
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.cells = nn.ModuleList([])
        lastHiddenDim = -1
        for i, hidden_dims in enumerate(self.hidden_structs):
            if i == 0:
                cell = nn.LSTMCell(hidden_dims * 2, hidden_dims)
            else:
                cell = nn.LSTMCell(lastHiddenDim, hidden_dims)

            #self.add_module("Cell_{}".format(i), cell)
            self.cells.append(cell)
            lastHiddenDim = hidden_dims
        self.tanh1 = nn.Tanh()
        # 注意力ta用这个
        self.conv1 = nn.Conv1d(in_channels=config.hidden_size*4, out_channels=config.hidden_size, kernel_size=3)
        self.pool1d = nn.MaxPool1d(kernel_size=config.pad_size)
        self.fc1 = nn.Linear(config.hidden_size*(1+2), config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(self.hidden_structs[-1], config.num_classes)


    def forward(self, x):
        x, _ = x
        embed_out = self.embedding(x)  # [batch_size, seq_len, embeding]=[16, 32, 300]
        H, _ = self.lstm(embed_out)
        #re_out = _rnn_reformat(H, self.hidden_structs[0] * 2, self.n_steps)
        #layer_outputs = self.multi_dRNN(re_out)
        #ta = torch.cat(layer_outputs, dim=1).reshape(self.batch_size, self.n_steps, self.hidden_structs[0])
        M = self.tanh1(H)  #     [batch_size, seq_len, hidden_size] =[16, 32, 128]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [batch, seq_len, 1]
        out = H * alpha  # [20, 32, 256] = [batch_size,seq_len,hidden]
        cat_out = self.cnns_concat(out,self.hidden_structs[0] * 2,self.n_steps)
        output = self.conv1(cat_out)
        re_lstm = H.permute(0,2,1)
        c_out = torch.cat((output, re_lstm), 1)
        pool1d_value = self.pool1d(c_out).squeeze()
        out = self.fc1(pool1d_value)
        pred = self.linear(out)

        return pred

    def cnns_concat(self,x,input_dim,n_steps):
        re_out = _rnn_reformat(x,input_dim,n_steps) # a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
        # 新建一个tensor shape(batch_size,input_dim)，全部0填充
        re_out = list(re_out)

        zero_tensor = torch.zeros_like(re_out[0])
        result = []
        length = len(re_out)
        re_out.append(zero_tensor)
        re_out.insert(0, zero_tensor)
        for i in range(length):
            t = torch.cat((re_out[i], re_out[i + 2]), 1)
            result.append(t)
        #尾部填充2个step 把形状还回n_step
        s1 = torch.cat((re_out[0],re_out[0]),1)
        result.append(s1)
        result.append(s1)
        ta = torch.cat(result, dim=1).reshape(self.batch_size, n_steps+2, input_dim*2)
        tb = ta.permute(0, 2, 1) #tb = [batch_size,hiden_size,n_steps]
        return tb


    def dRNN(self, cell, inputs, rate):
        """
        function: 此函数用于生成一层Dilated RNN
        @param cell: RNN cell
        @param inputs: shape: (n_steps, batch_size, input_dims) a list of n_steps tensors,each shape batch_size,input_dims
        @param rate: dilation rate
        @return: dilated RNN output
        """

        n_steps = len(inputs) #根据inputs的结构，长度即为 n_steps,也就是seq_len
        batch_size = inputs[0].size()[0]
        hidden_size = cell.hidden_size
        if rate < 0 or rate >= n_steps:
            raise ValueError('The \'rate\' variable needs to be adjusted.')
        msg1 = "Building layer: {0}, input length: {1}, dilation rate: {2}, input dim: {3}."
        #print(msg1.format('dilated', n_steps, rate, inputs[0].size()[1]))
        # 判断步数与dilation rate之间是否能整除，否则进行0填充
        FLAG = (n_steps % rate) == 0
        if not FLAG:
            #新建一个tensor shape(batch_size,input_dim)，全部0填充
            zero_tensor = torch.zeros_like(inputs[0])
            dilated_n_steps = n_steps // rate + 1 #输入数据分为rate组，没组dilated_n_steps个时间步
            for i_pad in range(dilated_n_steps * rate - n_steps):
                inputs.append(zero_tensor) # 在尾部填充全0 tensor
        else:
            # 能整除
            dilated_n_steps = n_steps // rate

        # now the length of 'inputs' divide rate
        # reshape it in the format of a list of tensors
        # the length of the list is 'dialated_n_steps'
        # the shape of each tensor is [batch_size * rate, input_dims]
        # by stacking tensors that "colored" the same

        # Example:
        # n_steps is 5, rate is 2, inputs = [x1, x2, x3, x4, x5]
        # zero-padding --> [x1, x2, x3, x4, x5, 0]
        # we want to have --> [[x1; x2], [x3; x4], [x_5; 0]]
        # which the length is the ceiling of n_steps/rate

        # 产生的每一项的数据格式为 tensor([[1., 1., 1.],
        #                       [1., 1., 1.],
        #                       [1., 1., 1.],
        #                       [1., 1., 1.]])

        dilated_inputs = [torch.cat([inputs[i * rate + j] for j in range(rate)], dim=0) for i in range(dilated_n_steps)]

        # 产生的每一项的数据格式为 tensor([[[1., 1., 1.],
        #                               [1., 1., 1.]],
        #                               [[1., 1., 1.],
        #                               [1., 1., 1.]]])
        # dilated_inputs = [torch.cat(torch.split(inputs[i * rate: (i + 1) * rate], rate, dim=0), dim=0) for i in range(dilated_n_steps)]
        dilated_outputs = []
        #hidden, cstate = self.init_hidden(batch_size * rate, hidden_size)
        for dilated_input in dilated_inputs:
            #dilated_input = dilated_input.type(torch.FloatTensor)
            hidden, cstate = cell(dilated_input)
            dilated_outputs.append(hidden)
        # cell输出的hidden shape(batch_size,hidden_size) dilated_outputs shape(dilated_n_steps,batch_size*rate,hidden_size)

        # reshape output back to the input format as a list of tensors with shape [batch_size, input_dims]
        # split each element of the outputs from size [batch_size*rate, input_dims] to
        # [[batch_size, input_dims], [batch_size, input_dims], ...] with length = rate
        splitted_outputs = [torch.chunk(output, rate, 0) for output in dilated_outputs]
        unrolled_outputs = [output for sublist in splitted_outputs for output in sublist]
        # remove padded zeros
        outputs = unrolled_outputs[:n_steps]
        # outputs shape(n_steps,batch_size,hidden_size) 暂时不知道上述转成input_dim的注释是否有问题
        return outputs

    def multi_dRNN(self, inputs):
        """
        Inputs:
            inputs -- A list of 'n_steps' tensors, each has shape (batch_size, input_dims).
        Outputs:
            outputs -- A list of 'n_steps' tensors, as the outputs for the top layer of the multi-dRNN.
        """
        assert (len(self.cells) == len(self.dilations))
        x = copy.copy(inputs)
        for cell, dilation in zip(self.cells, self.dilations):
            x = self.dRNN(cell, x, dilation)

        return x

    def init_hidden(self, batch_size, hidden_dim):
        hx = autograd.Variable(torch.zeros(batch_size, hidden_dim))
        cx = autograd.Variable(torch.zeros(batch_size, hidden_dim))
        hx = hx.to(self.device)
        cx = cx.to(self.device)
        return hx, cx




