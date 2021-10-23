import torch
import logging
from torch import nn
import torch.nn.functional as F
from tcan_block import TemporalConvNet
# from model.pe import PositionEmbedding
# from model.optimizations import VariationalDropout, WeightDropout

from IPython import embed


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class TCANet(nn.Module):

    def __init__(self, emb_size=600, input_output_size=5537, num_channels=2400, seq_len=80, num_sub_blocks=2, temp_attn=True, nheads=1, en_res=True,
                 conv=True, key_size=600, kernel_size=2, dropout=0.3, wdrop=0.0, emb_dropout=0.1, tied_weights=False, 
                 dataset_name=None, visual=True):
        super(TCANet, self).__init__()

        self.temp_attn = temp_attn
        self.dataset_name = dataset_name
        self.num_levels = num_channels
        self.word_encoder = nn.Embedding(input_output_size, emb_size)
        self.tcanet = TemporalConvNet(input_output_size, emb_size, num_channels, \
            num_sub_blocks, temp_attn, nheads, en_res, conv, key_size, kernel_size, visual=visual, dropout=dropout)
        self.drop = nn.Dropout(emb_dropout)
        self.decoder = nn.Linear(num_channels[-1], input_output_size)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.word_encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def get_conv_names(self, num_channels):
        conv_names_list = []
        for level_i in range(len(num_channels)):
            conv_names_list.append(['network', level_i, 'net', 0, 'weight_v'])
            conv_names_list.append(['network', level_i, 'net', 4, 'weight_v'])
        return conv_names_list

    def forward(self, input):

        emb = self.drop(self.word_encoder(input))
        if self.temp_attn:
            y, attn_weight_list = self.tcanet(emb.transpose(1, 2))
            y = self.decoder(y.transpose(1, 2))
            return y.contiguous(), [attn_weight_list[0], attn_weight_list[self.num_levels//2], attn_weight_list[-1]]
        else:
            y = self.tcanet(emb.transpose(1, 2))
            y = self.decoder(y.transpose(1, 2))
            return y.contiguous()