# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
# import Models.Transunet.networks.vit_seg_configs as configs
# from Models.Transunet.networks.vit_seg_modeling_resnet_skip import ResNetV2
import vit_seg_configs as configs
from vit_seg_modeling_resnet_skip import ResNetV2

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.patch_embeddings_b32 = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=(2,2),
                                       stride=(2,2))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.position_embeddings_b32 = nn.Parameter(torch.zeros(1, int(n_patches/4), config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.dropout_b32 = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x_b32 = x
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x_b32 = self.patch_embeddings_b32(x_b32)
        x = x.flatten(2)
        x_b32 = x_b32.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        x_b32 = x_b32.transpose(-1, -2)
        #print(x.size())
        #print(self.position_embeddings.size())

        embeddings = x + self.position_embeddings
        embeddings_b32 = x_b32+ self.position_embeddings_b32
        embeddings = self.dropout(embeddings)
        embeddings_b32 = self.dropout_b32(embeddings_b32)
        return embeddings,embeddings_b32, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        encoded = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            encoding = self.encoder_norm(hidden_states)
            encoded.append(encoding)
            if self.vis:
                attn_weights.append(weights)
        #encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Features_cross(nn.Module):
    def __init__(self, config, vis):
        super(Features_cross, self).__init__()
        self.vis = vis
        self.cross = Block(config, vis)
        #self.cross_1 = Block(config, vis)
        #self.cross_2 = Block(config, vis)
        self.cross_norm = LayerNorm(config.hidden_size, eps=1e-6)
        #self.cross_norm_1 = LayerNorm(config.hidden_size, eps=1e-6)
        #self.cross_norm_2 = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):

        crossing, weights = self.cross(hidden_states)
        crossing = self.cross_norm(crossing)

        return crossing





class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)
        self.encoder_b32 = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, embedding_output_b32, features = self.embeddings(input_ids)
        
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        encoded_b32, attn_weights = self.encoder_b32(embedding_output_b32)  # (B, n_patch, hidden)
        return encoded,encoded_b32, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        super().__init__(conv2d)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        #x = self.up(x)                                       #########################################
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x
    



class DecoderCup_b32(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        x = self.up(x)                                       #########################################
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x









class Multioutput(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.decoder = DecoderCup(config)

    def forward(self, hidden_states,features):
        
        x = self.decoder(hidden_states, features)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=304, num_classes=2, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)


        self.Features_cross_0 = Features_cross(config, vis)
        self.Features_cross_1 = Features_cross(config, vis)
        self.Features_cross_2 = Features_cross(config, vis)
        self.Features_cross_3 = Features_cross(config, vis)
        self.Features_cross_4 = Features_cross(config, vis)
        self.Features_cross_5 = Features_cross(config, vis)
        
        #self.decoder = DecoderCup(config)

        self.Multioutput_0 = Multioutput(config)
        self.Multioutput_1_faz = Multioutput(config)
        self.Multioutput_2_faz = Multioutput(config)
        self.Multioutput_3_faz = Multioutput(config)
        self.Multioutput_4_faz = Multioutput(config)
        self.Multioutput_5_faz = Multioutput(config)
        #self.Multioutput_6_faz = Multioutput(config)

        self.Multioutput_0_cross = Multioutput(config)
        self.Multioutput_1_faz_cross = Multioutput(config)
        self.Multioutput_2_faz_cross = Multioutput(config)
        self.Multioutput_3_faz_cross = Multioutput(config)
        self.Multioutput_4_faz_cross = Multioutput(config)
        self.Multioutput_5_faz_cross = Multioutput(config)
        #self.Multioutput_6_faz_cross = Multioutput(config)

        self.DecoderCup_b32_0 = DecoderCup_b32(config)
        self.DecoderCup_b32_faz_1 = DecoderCup_b32(config)
        self.DecoderCup_b32_faz_2 = DecoderCup_b32(config)
        self.DecoderCup_b32_faz_3 = DecoderCup_b32(config)
        self.DecoderCup_b32_faz_4 = DecoderCup_b32(config)
        self.DecoderCup_b32_faz_5 = DecoderCup_b32(config)
        #self.DecoderCup_b32_faz_6 = DecoderCup_b32(config)

        self.DecoderCup_b32_0_cross = DecoderCup_b32(config)
        self.DecoderCup_b32_faz_1_cross = DecoderCup_b32(config)
        self.DecoderCup_b32_faz_2_cross = DecoderCup_b32(config)
        self.DecoderCup_b32_faz_3_cross = DecoderCup_b32(config)
        self.DecoderCup_b32_faz_4_cross = DecoderCup_b32(config)
        self.DecoderCup_b32_faz_5_cross = DecoderCup_b32(config)
        #self.DecoderCup_b32_faz_6_cross = DecoderCup_b32(config)
        
        self.segmentation_head_faz_2 = SegmentationHead(
            in_channels=20,                                                     #12
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.segmentation_head_2 = SegmentationHead(
            in_channels=4,                                                    #12
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, x_b32, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        

        
        temp = torch.cat([x[0],x_b32[0]], 1)
        temp = self.Features_cross_0(temp)
        x.append(temp[:,0:x[0].size()[1],:])
        x_b32.append(temp[:,x[0].size()[1]:temp.size()[1],:])

        temp = torch.cat([x[1],x_b32[1]], 1)
        temp = self.Features_cross_1(temp)
        x.append(temp[:,0:x[1].size()[1],:])
        x_b32.append(temp[:,x[1].size()[1]:temp.size()[1],:])

        temp = torch.cat([x[2],x_b32[2]], 1)
        temp = self.Features_cross_2(temp)
        x.append(temp[:,0:x[2].size()[1],:])
        x_b32.append(temp[:,x[2].size()[1]:temp.size()[1],:])

        temp = torch.cat([x[3],x_b32[3]], 1)
        temp = self.Features_cross_3(temp)
        x.append(temp[:,0:x[3].size()[1],:])
        x_b32.append(temp[:,x[3].size()[1]:temp.size()[1],:])

        temp = torch.cat([x[4],x_b32[4]], 1)
        temp = self.Features_cross_4(temp)
        x.append(temp[:,0:x[4].size()[1],:])
        x_b32.append(temp[:,x[4].size()[1]:temp.size()[1],:])

        temp = torch.cat([x[5],x_b32[5]], 1)
        temp = self.Features_cross_5(temp)
        x.append(temp[:,0:x[5].size()[1],:])
        x_b32.append(temp[:,x[5].size()[1]:temp.size()[1],:])
            

        print('x:',x[0].size(),len(x))
        print('x_b32:', x_b32[0].size(), len(x_b32))


        result = self.Multioutput_0(x[0],features)
        result = torch.cat([result, self.DecoderCup_b32_0(x_b32[0],features)], 1)
        #result = torch.cat([result, self.Multioutput_1_faz(x[1],features)], 1)
        #result = torch.cat([result, self.DecoderCup_b32_faz_1(x_b32[1],features)], 1)

        result = torch.cat([result, self.DecoderCup_b32_0_cross(x_b32[6],features)], 1)
        result = torch.cat([result, self.Multioutput_0_cross(x[6],features)], 1)
        #result = torch.cat([result, self.DecoderCup_b32_faz_1_cross(x_b32[8],features)], 1)
        #result = torch.cat([result, self.Multioutput_1_faz_cross(x[8],features)], 1)

        logits = self.segmentation_head_2(result)

        result_faz = self.Multioutput_1_faz(x[1],features)
        result_faz = torch.cat([result_faz, self.Multioutput_2_faz(x[2],features)], 1)
        result_faz = torch.cat([result_faz, self.Multioutput_3_faz(x[3],features)], 1)
        result_faz = torch.cat([result_faz, self.Multioutput_4_faz(x[4],features)], 1)
        result_faz = torch.cat([result_faz, self.Multioutput_5_faz(x[5],features)], 1)

        result_faz = torch.cat([result_faz, self.DecoderCup_b32_faz_1(x_b32[1], features)], 1)
        result_faz = torch.cat([result_faz, self.DecoderCup_b32_faz_2(x_b32[2], features)], 1)
        result_faz = torch.cat([result_faz, self.DecoderCup_b32_faz_3(x_b32[3], features)], 1)
        result_faz = torch.cat([result_faz, self.DecoderCup_b32_faz_4(x_b32[4], features)], 1)
        result_faz = torch.cat([result_faz, self.DecoderCup_b32_faz_5(x_b32[5], features)], 1)

        result_faz = torch.cat([result_faz, self.Multioutput_1_faz_cross(x[7],features)], 1)
        result_faz = torch.cat([result_faz, self.Multioutput_2_faz_cross(x[8],features)], 1)
        result_faz = torch.cat([result_faz, self.Multioutput_3_faz_cross(x[9],features)], 1)
        result_faz = torch.cat([result_faz, self.Multioutput_4_faz_cross(x[10],features)], 1)
        result_faz = torch.cat([result_faz, self.Multioutput_5_faz_cross(x[11],features)], 1)

        result_faz = torch.cat([result_faz, self.DecoderCup_b32_faz_1_cross(x_b32[7], features)], 1)
        result_faz = torch.cat([result_faz, self.DecoderCup_b32_faz_2_cross(x_b32[8], features)], 1)
        result_faz = torch.cat([result_faz, self.DecoderCup_b32_faz_3_cross(x_b32[9], features)], 1)
        result_faz = torch.cat([result_faz, self.DecoderCup_b32_faz_4_cross(x_b32[10], features)], 1)
        result_faz = torch.cat([result_faz, self.DecoderCup_b32_faz_5_cross(x_b32[11], features)], 1)
        
        
        logits_faz = self.segmentation_head_faz_2(result_faz)
        logits = torch.sigmoid(logits)
        logits_faz = torch.sigmoid(logits_faz)                                           #logits_faz
        #logits = logits_faz
        return logits ,logits_faz

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


