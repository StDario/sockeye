# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Dict, Optional

import mxnet as mx
import numpy as np

from . import config
from . import constants as C
from . import layers
from typing import List


class TransformerConfig(config.Config):

    def __init__(self,
                 model_size: int,
                 attention_heads: int,
                 feed_forward_num_hidden: int,
                 act_type: str,
                 num_layers: int,
                 dropout_attention: float,
                 dropout_act: float,
                 dropout_prepost: float,
                 positional_embedding_type: str,
                 preprocess_sequence: str,
                 postprocess_sequence: str,
                 max_seq_len_source: int,
                 max_seq_len_target: int,
                 model_type: str,
                 conv_config: Optional['ConvolutionalEmbeddingConfig'] = None) -> None:  # type: ignore
        super().__init__()
        self.model_size = model_size
        self.attention_heads = attention_heads
        self.feed_forward_num_hidden = feed_forward_num_hidden
        self.act_type = act_type
        self.num_layers = num_layers
        self.dropout_attention = dropout_attention
        self.dropout_act = dropout_act
        self.dropout_prepost = dropout_prepost
        self.positional_embedding_type = positional_embedding_type
        self.preprocess_sequence = preprocess_sequence
        self.postprocess_sequence = postprocess_sequence
        self.max_seq_len_source = max_seq_len_source
        self.max_seq_len_target = max_seq_len_target
        self.conv_config = conv_config
        self.model_type = model_type

class TransformerEncoderBlock:
    """
    A transformer encoder block consists self-attention and a feed-forward layer with pre/post process blocks
    in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 prefix: str,
                 model_type: str = 'ctx_dec',
                 num_layer: int = 0) -> None:

        self.model_type = model_type
        self.num_layer = num_layer
        self.pre_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_self_pre_" % prefix)
        self.self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%satt_self_" % prefix)
        self.post_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           num_hidden=config.model_size,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%satt_self_post_" % prefix)


        self.pre_ctx_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%sctx_att_self_pre_" % prefix)
        self.ctx_self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%sctx_att_self_" % prefix)
        self.post_ctx_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           num_hidden=config.model_size,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%sctx_att_self_post_" % prefix)

        self.pre_ctx_mh_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                              num_hidden=config.model_size,
                                                              dropout=config.dropout_prepost,
                                                              prefix="%sctx_mh_att_self_pre_" % prefix)
        self.ctx_mh_attention = layers.MultiHeadAttention(depth_att=config.model_size,
                                                                heads=config.attention_heads,
                                                                depth_out=config.model_size,
                                                                dropout=config.dropout_attention,
                                                                prefix="%sctx_mh_att_self_" % prefix)
        self.post_ctx_mh_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                               num_hidden=config.model_size,
                                                               dropout=config.dropout_prepost,
                                                               prefix="%sctx_mh_att_self_post_" % prefix)


        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              num_hidden=config.model_size,
                                              dropout=config.dropout_prepost,
                                              prefix="%sff_pre_" % prefix)
        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sff_" % prefix)
        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               num_hidden=config.model_size,
                                               dropout=config.dropout_prepost,
                                               prefix="%sff_post_" % prefix)


        self.ctx_pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              num_hidden=config.model_size,
                                              dropout=config.dropout_prepost,
                                              prefix="%sctx_ff_pre_" % prefix)
        self.ctx_ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sctx_ff_" % prefix)

        self.ctx_post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               num_hidden=config.model_size,
                                               dropout=config.dropout_prepost,
                                               prefix="%sctx_ff_post_" % prefix)

    def __call__(self, data: mx.sym.Symbol, bias: mx.sym.Symbol, ctx_data: mx.sym.Symbol, ctx_bias: mx.sym.Symbol) -> (mx.sym.Symbol, mx.sym.Symbol):
        # self-attention

        data_self_att = self.self_attention(inputs=self.pre_self_attention(data, None),
                                            bias=bias,
                                            cache=None)
        data = self.post_self_attention(data_self_att, data)

        # feed-forward
        data_ff = self.ff(self.pre_ff(data, None))
        data = self.post_ff(data_ff, data)

        if self.model_type == "ctx_dec":
            ctx_data_self_att = self.ctx_self_attention(inputs=self.pre_ctx_self_attention(ctx_data, None),
                                                bias=ctx_bias,
                                                cache=None)
            ctx_data = self.post_ctx_self_attention(ctx_data_self_att, ctx_data)

        elif self.model_type == "ctx_dec_alt":

            if self.num_layer % 2 == 0:
                ctx_data_self_att = self.ctx_self_attention(inputs=self.pre_ctx_self_attention(ctx_data, None),
                                                            bias=ctx_bias,
                                                            cache=None)
                ctx_data = self.post_ctx_self_attention(ctx_data_self_att, ctx_data)
            else:
                ctx_data_att = self.ctx_mh_attention(queries=self.pre_ctx_mh_attention(ctx_data, None), memory=data,
                                                     bias=bias)
                ctx_data = self.post_ctx_mh_attention(ctx_data_att, ctx_data)

        ctx_data_ff = self.ctx_ff(self.ctx_pre_ff(ctx_data, None))
        ctx_data = self.ctx_post_ff(ctx_data_ff, ctx_data)

        return data, ctx_data

class TransformerGateEncoderBlock:
    """
    A transformer encoder block consists self-attention and a feed-forward layer with pre/post process blocks
    in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 prefix: str) -> None:

        self.pre_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_self_pre_" % prefix)
        self.self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%satt_self_" % prefix)
        self.post_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           num_hidden=config.model_size,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%satt_self_post_" % prefix)


        self.pre_ctx_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%sctx_att_self_pre_" % prefix)
        self.ctx_self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%sctx_att_self_" % prefix)
        self.post_ctx_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           num_hidden=config.model_size,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%sctx_att_self_post_" % prefix)

        self.pre_ctx_mh_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                              num_hidden=config.model_size,
                                                              dropout=config.dropout_prepost,
                                                              prefix="%sctx_mh_att_self_pre_" % prefix)
        self.ctx_mh_attention = layers.MultiHeadAttention(depth_att=config.model_size,
                                                                heads=config.attention_heads,
                                                                depth_out=config.model_size,
                                                                dropout=config.dropout_attention,
                                                                prefix="%sctx_mh_att_self_" % prefix)
        self.post_ctx_mh_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                               num_hidden=config.model_size,
                                                               dropout=config.dropout_prepost,
                                                               prefix="%sctx_mh_att_self_post_" % prefix)


        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              num_hidden=config.model_size,
                                              dropout=config.dropout_prepost,
                                              prefix="%sff_pre_" % prefix)
        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sff_" % prefix)
        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               num_hidden=config.model_size,
                                               dropout=config.dropout_prepost,
                                               prefix="%sff_post_" % prefix)


        self.ctx_pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              num_hidden=config.model_size,
                                              dropout=config.dropout_prepost,
                                              prefix="%sctx_ff_pre_" % prefix)
        self.ctx_ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sctx_ff_" % prefix)

        self.ctx_post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               num_hidden=config.model_size,
                                               dropout=config.dropout_prepost,
                                               prefix="%sctx_ff_post_" % prefix)

        self.gate = EncoderGateBlock(num_hidden=config.model_size, dropout=config.dropout_act, act_type=config.act_type,
                              prefix="%senc_gate" % prefix)

    def __call__(self, data: mx.sym.Symbol, bias: mx.sym.Symbol, ctx_data: mx.sym.Symbol, ctx_bias: mx.sym.Symbol) -> (mx.sym.Symbol, mx.sym.Symbol):
        # self-attention

        data_self_att = self.self_attention(inputs=self.pre_self_attention(data, None),
                                            bias=bias,
                                            cache=None)
        data = self.post_self_attention(data_self_att, data)

        ctx_data_self_att = self.ctx_self_attention(inputs=self.pre_ctx_self_attention(ctx_data, None),
                                            bias=ctx_bias,
                                            cache=None)
        ctx_data = self.post_ctx_self_attention(ctx_data_self_att, ctx_data)

        ctx_data_ff = self.ctx_ff(self.ctx_pre_ff(ctx_data, None))
        ctx_data = self.ctx_post_ff(ctx_data_ff, ctx_data)



        ctx_data_att = self.ctx_mh_attention(queries=self.pre_ctx_mh_attention(data, None), memory=ctx_data,
                                               bias=ctx_bias)
        ctx_data = self.post_ctx_mh_attention(ctx_data_att, data)


        data = self.gate(data, ctx_data)


        # feed-forward
        data_ff = self.ff(self.pre_ff(data, None))
        data = self.post_ff(data_ff, data)

        return data, ctx_data

class TransformerEncoderBlockShared:
    """
    A transformer encoder block consists self-attention and a feed-forward layer with pre/post process blocks
    in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 prefix: str, model_type: str = 'ctx_dec',
                 num_layer: int = 0) -> None:

        self.model_type = model_type
        self.num_layer = num_layer
        self.pre_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_self_pre_" % prefix)
        self.self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%satt_self_" % prefix)
        self.post_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           num_hidden=config.model_size,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%satt_self_post_" % prefix)


        self.pre_ctx_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%sctx_att_self_pre_" % prefix)
        self.ctx_self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%sctx_att_self_" % prefix)
        self.post_ctx_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           num_hidden=config.model_size,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%sctx_att_self_post_" % prefix)


        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              num_hidden=config.model_size,
                                              dropout=config.dropout_prepost,
                                              prefix="%sff_pre_" % prefix)
        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sff_" % prefix)
        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               num_hidden=config.model_size,
                                               dropout=config.dropout_prepost,
                                               prefix="%sff_post_" % prefix)

        self.pre_ctx_mh_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                            num_hidden=config.model_size,
                                                            dropout=config.dropout_prepost,
                                                            prefix="%sctx_mh_att_shared_self_pre_" % prefix)
        self.ctx_mh_attention = layers.MultiHeadAttention(depth_att=config.model_size,
                                                          heads=config.attention_heads,
                                                          depth_out=config.model_size,
                                                          dropout=config.dropout_attention,
                                                          prefix="%sctx_mh_att_shared_self_" % prefix)
        self.post_ctx_mh_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                             num_hidden=config.model_size,
                                                             dropout=config.dropout_prepost,
                                                             prefix="%sctx_mh_att_shared_self_post_" % prefix)

    def __call__(self, data: mx.sym.Symbol, bias: mx.sym.Symbol, ctx_data: mx.sym.Symbol, ctx_bias: mx.sym.Symbol) -> (mx.sym.Symbol, mx.sym.Symbol):
        # self-attention

        data_self_att = self.self_attention(inputs=self.pre_self_attention(data, None),
                                            bias=bias,
                                            cache=None)
        data = self.post_self_attention(data_self_att, data)

        if self.model_type == "ctx_dec" or self.model_type == "ctx_enc":
            ctx_data_self_att = self.self_attention(inputs=self.pre_self_attention(ctx_data, None),
                                                bias=ctx_bias,
                                                cache=None)
            ctx_data = self.post_self_attention(ctx_data_self_att, ctx_data)

        elif self.model_type == "ctx_dec_alt":
            if self.num_layer % 2 == 0:
                ctx_data_self_att = self.self_attention(inputs=self.pre_self_attention(ctx_data, None),
                                                            bias=ctx_bias,
                                                            cache=None)
                ctx_data = self.post_self_attention(ctx_data_self_att, ctx_data)
            else:
                ctx_data_att = self.ctx_mh_attention(queries=self.pre_ctx_mh_attention(ctx_data, None), memory=data,
                                                     bias=bias)
                ctx_data = self.post_ctx_mh_attention(ctx_data_att, ctx_data)


        # feed-forward
        data_ff = self.ff(self.pre_ff(data, None))
        data = self.post_ff(data_ff, data)


        ctx_data_ff = self.ff(self.pre_ff(ctx_data, None))
        ctx_data = self.post_ff(ctx_data_ff, ctx_data)

        return data, ctx_data

class TransformerCtxStandardEncoderBlock:
    """
    A transformer encoder block consists self-attention and a feed-forward layer with pre/post process blocks
    in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 prefix: str, model_type: str = 'ctx_dec',
                 num_layer: int = 0) -> None:

        self.model_type = model_type
        self.num_layer = num_layer
        self.pre_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_self_pre_" % prefix)
        self.self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%satt_self_" % prefix)
        self.post_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           num_hidden=config.model_size,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%satt_self_post_" % prefix)


        self.pre_ctx_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%sctx_att_self_pre_" % prefix)
        self.ctx_self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%sctx_att_self_" % prefix)
        self.post_ctx_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           num_hidden=config.model_size,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%sctx_att_self_post_" % prefix)


        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              num_hidden=config.model_size,
                                              dropout=config.dropout_prepost,
                                              prefix="%sff_pre_" % prefix)
        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sff_" % prefix)
        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               num_hidden=config.model_size,
                                               dropout=config.dropout_prepost,
                                               prefix="%sff_post_" % prefix)


        self.pre_ctx_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              num_hidden=config.model_size,
                                              dropout=config.dropout_prepost,
                                              prefix="%sctx_ff_pre_" % prefix)

        self.ctx_ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sctx_ff_" % prefix)

        self.post_ctx_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               num_hidden=config.model_size,
                                               dropout=config.dropout_prepost,
                                               prefix="%sctx_ff_post_" % prefix)

    def __call__(self, data: mx.sym.Symbol, bias: mx.sym.Symbol, ctx_data: mx.sym.Symbol, ctx_bias: mx.sym.Symbol) -> (mx.sym.Symbol, mx.sym.Symbol):
        # self-attention

        data_self_att = self.self_attention(inputs=self.pre_self_attention(data, None),
                                            bias=bias,
                                            cache=None)
        data = self.post_self_attention(data_self_att, data)

        ctx_data_self_att = self.self_attention(inputs=self.pre_self_attention(ctx_data, None),
                                            bias=ctx_bias,
                                            cache=None)
        ctx_data = self.post_self_attention(ctx_data_self_att, ctx_data)


        # feed-forward
        data_ff = self.ff(self.pre_ff(data, None))
        data = self.post_ff(data_ff, data)


        ctx_data_ff = self.ctx_ff(self.pre_ctx_ff(ctx_data, None))
        ctx_data = self.post_ctx_ff(ctx_data_ff, ctx_data)

        return data, ctx_data

class TransformerDecoderBlock:
    """
    A transformer encoder block consists self-attention, encoder attention, and a feed-forward layer
    with pre/post process blocks in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 prefix: str) -> None:

        self.prefix = prefix
        self.pre_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_self_pre_" % prefix)
        self.self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%satt_self_" % prefix)
        self.post_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           num_hidden=config.model_size,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%satt_self_post_" % prefix)

        self.pre_enc_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                         num_hidden=config.model_size,
                                                         dropout=config.dropout_prepost,
                                                         prefix="%satt_enc_pre_" % prefix)
        self.enc_attention = layers.MultiHeadAttention(depth_att=config.model_size,
                                                       heads=config.attention_heads,
                                                       depth_out=config.model_size,
                                                       dropout=config.dropout_attention,
                                                       prefix="%satt_enc_" % prefix)
        self.post_enc_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_enc_post_" % prefix)

        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              num_hidden=config.model_size,
                                              dropout=config.dropout_prepost,
                                              prefix="%sff_pre_" % prefix)
        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sff_" % prefix)
        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               num_hidden=config.model_size,
                                               dropout=config.dropout_prepost,
                                               prefix="%sff_post_" % prefix)

    def __call__(self,
                 target: mx.sym.Symbol,
                 target_bias: mx.sym.Symbol,
                 source: mx.sym.Symbol,
                 source_bias: mx.sym.Symbol,
                 ctx_source: mx.sym.Symbol,
                 ctx_source_bias: mx.sym.Symbol,
                 cache: Optional[Dict[str, Optional[mx.sym.Symbol]]] = None) -> mx.sym.Symbol:

        target_self_att = self.self_attention(inputs=self.pre_self_attention(target, None),
                                              bias=target_bias,
                                              cache=cache)
        target = self.post_self_attention(target_self_att, target)

        # encoder attention
        target_enc_att = self.enc_attention(queries=self.pre_enc_attention(target, None),
                                            memory=source,
                                            bias=source_bias)
        target = self.post_enc_attention(target_enc_att, target)

        # feed-forward
        target_ff = self.ff(self.pre_ff(target, None))
        target = self.post_ff(target_ff, target)



        return target


class TransformerCtxComplexDecoderBlock:
    """
    A transformer encoder block consists self-attention, encoder attention, and a feed-forward layer
    with pre/post process blocks in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 prefix: str) -> None:

        self.prefix = prefix
        self.pre_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_self_pre_" % prefix)
        self.self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%satt_self_" % prefix)
        self.post_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           num_hidden=config.model_size,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%satt_self_post_" % prefix)

        self.pre_enc_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                         num_hidden=config.model_size,
                                                         dropout=config.dropout_prepost,
                                                         prefix="%satt_enc_pre_" % prefix)
        self.enc_attention = layers.MultiHeadAttention(depth_att=config.model_size,
                                                       heads=config.attention_heads,
                                                       depth_out=config.model_size,
                                                       dropout=config.dropout_attention,
                                                       prefix="%satt_enc_" % prefix)
        self.post_enc_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_enc_post_" % prefix)

        self.pre_ctx_enc_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                             num_hidden=config.model_size,
                                                             dropout=config.dropout_prepost,
                                                             prefix="%sctx_att_enc_pre_" % prefix)
        self.ctx_enc_attention = layers.MultiHeadAttention(depth_att=config.model_size,
                                                           heads=config.attention_heads,
                                                           depth_out=config.model_size,
                                                           dropout=config.dropout_attention,
                                                           prefix="%sctx_att_enc_" % prefix)
        self.post_ctx_enc_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                              num_hidden=config.model_size,
                                                              dropout=config.dropout_prepost,
                                                              prefix="%sctx_att_enc_post_" % prefix)

        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              num_hidden=config.model_size * 2,
                                              dropout=config.dropout_prepost,
                                              prefix="%sff_pre_" % prefix)

        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sff_" % prefix)

        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               num_hidden=config.model_size,
                                               dropout=config.dropout_prepost,
                                               prefix="%sff_post_" % prefix)

        self.pre_ctx_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                  num_hidden=config.model_size * 2,
                                                  dropout=config.dropout_prepost,
                                                  prefix="%sff_pre_ctx_" % prefix)

        self.ctx_ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                             num_model=config.model_size,
                                             act_type=config.act_type,
                                             dropout=config.dropout_act,
                                             prefix="%sctx_ff_" % prefix)

        self.post_ctx_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                   num_hidden=config.model_size,
                                                   dropout=config.dropout_prepost,
                                                   prefix="%sff_post_ctx_" % prefix)

        self.pre_preattn_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                      num_hidden=config.model_size * 2,
                                                      dropout=config.dropout_prepost,
                                                      prefix="%sff_pre_att_" % prefix)
        self.preattn_ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                                 num_model=config.model_size,
                                                 act_type=config.act_type,
                                                 dropout=config.dropout_act,
                                                 prefix="%sff_preattn_" % prefix)

        self.post_preattn_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                       num_hidden=config.model_size,
                                                       dropout=config.dropout_prepost,
                                                       prefix="%sff_post_preattn_" % prefix)

        self.gate = GateTrgBlock(num_hidden=config.model_size, dropout=config.dropout_act, act_type=config.act_type,
                              prefix="%sgate" % prefix)

        self.pre_postgate_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                       num_hidden=config.model_size,
                                                       dropout=config.dropout_prepost,
                                                       prefix="%sff_pre_postgate_" % prefix)
        self.postgate_ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                                  num_model=config.model_size,
                                                  act_type=config.act_type,
                                                  dropout=config.dropout_act,
                                                  prefix="%sff_postgate_" % prefix)

        self.post_postgate_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                        num_hidden=config.model_size,
                                                        dropout=config.dropout_prepost,
                                                        prefix="%sff_post_postgate_" % prefix)

    def __call__(self,
                 target: mx.sym.Symbol,
                 target_bias: mx.sym.Symbol,
                 source: mx.sym.Symbol,
                 source_bias: mx.sym.Symbol,
                 ctx_source: mx.sym.Symbol,
                 ctx_source_bias: mx.sym.Symbol,
                 cache: Optional[Dict[str, Optional[mx.sym.Symbol]]] = None) -> mx.sym.Symbol:
        # self-attention
        target_self_att = self.self_attention(inputs=self.pre_self_attention(target, None),
                                              bias=target_bias,
                                              cache=cache)
        target = self.post_self_attention(target_self_att, target)

        target_aux = self.preattn_ff(self.pre_preattn_ff(target, None))
        target_aux = self.post_preattn_ff(target_aux, target)


        # encoder attention
        target_enc_att = self.enc_attention(queries=self.pre_enc_attention(target, None),
                                            memory=source,
                                            bias=source_bias)
        target_main = self.post_enc_attention(target_enc_att, target)

        # ctx encoder attention
        target_ctx_enc_att = self.ctx_enc_attention(queries=self.pre_ctx_enc_attention(target_aux, None),
                                                    memory=ctx_source,
                                                    bias=ctx_source_bias)
        target_ctx = self.post_ctx_enc_attention(target_ctx_enc_att, target_aux)

        # target_main_ff = self.ff(self.pre_ff(target_main, None))
        # target_main = self.post_ff(target_main_ff, target_main)
        #
        # target_ctx_ff = self.ctx_ff(self.pre_ctx_ff(target_ctx, None))
        # target_ctx = self.post_ctx_ff(target_ctx_ff, target_ctx)


        target = self.gate(target_main, target_ctx, target)

        target_postgate_ff = self.postgate_ff(self.pre_postgate_ff(target, None))
        target = self.post_postgate_ff(target_postgate_ff, target)


        return target

class TransformerCtxDecoderBlock:
    """
    A transformer encoder block consists self-attention, encoder attention, and a feed-forward layer
    with pre/post process blocks in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 prefix: str) -> None:

        self.prefix = prefix
        self.pre_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_self_pre_" % prefix)
        self.self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%satt_self_" % prefix)
        self.post_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           num_hidden=config.model_size,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%satt_self_post_" % prefix)

        self.pre_enc_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                         num_hidden=config.model_size,
                                                         dropout=config.dropout_prepost,
                                                         prefix="%satt_enc_pre_" % prefix)
        self.enc_attention = layers.MultiHeadAttention(depth_att=config.model_size,
                                                       heads=config.attention_heads,
                                                       depth_out=config.model_size,
                                                       dropout=config.dropout_attention,
                                                       prefix="%satt_enc_" % prefix)
        self.post_enc_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_enc_post_" % prefix)

        self.pre_ctx_enc_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                         num_hidden=config.model_size,
                                                         dropout=config.dropout_prepost,
                                                         prefix="%sctx_att_enc_pre_" % prefix)
        self.ctx_enc_attention = layers.MultiHeadAttention(depth_att=config.model_size,
                                                       heads=config.attention_heads,
                                                       depth_out=config.model_size,
                                                       dropout=config.dropout_attention,
                                                       prefix="%sctx_att_enc_" % prefix)
        self.post_ctx_enc_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                          num_hidden=config.model_size,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%sctx_att_enc_post_" % prefix)

        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              num_hidden=config.model_size * 2,
                                              dropout=config.dropout_prepost,
                                              prefix="%sff_pre_" % prefix)

        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sff_" % prefix)

        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               num_hidden=config.model_size,
                                               dropout=config.dropout_prepost,
                                               prefix="%sff_post_" % prefix)


        self.pre_ctx_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              num_hidden=config.model_size * 2,
                                              dropout=config.dropout_prepost,
                                              prefix="%sff_pre_ctx_" % prefix)


        self.ctx_ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sctx_ff_" % prefix)

        self.post_ctx_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               num_hidden=config.model_size,
                                               dropout=config.dropout_prepost,
                                               prefix="%sff_post_ctx_" % prefix)

        self.pre_preattn_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              num_hidden=config.model_size * 2,
                                              dropout=config.dropout_prepost,
                                              prefix="%sff_pre_att_" % prefix)
        self.preattn_ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sff_preattn_" % prefix)

        self.post_preattn_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               num_hidden=config.model_size,
                                               dropout=config.dropout_prepost,
                                               prefix="%sff_post_preattn_" % prefix)

        self.gate = GateBlock(num_hidden=config.model_size, dropout=config.dropout_act, act_type=config.act_type, prefix="%sgate" % prefix)

        self.pre_postgate_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                      num_hidden=config.model_size,
                                                      dropout=config.dropout_prepost,
                                                      prefix="%sff_pre_postgate_" % prefix)
        self.postgate_ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                                 num_model=config.model_size,
                                                 act_type=config.act_type,
                                                 dropout=config.dropout_act,
                                                 prefix="%sff_postgate_" % prefix)

        self.post_postgate_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                       num_hidden=config.model_size,
                                                       dropout=config.dropout_prepost,
                                                       prefix="%sff_post_postgate_" % prefix)

    def __call__(self,
                 target: mx.sym.Symbol,
                 target_bias: mx.sym.Symbol,
                 source: mx.sym.Symbol,
                 source_bias: mx.sym.Symbol,
                 ctx_source: mx.sym.Symbol,
                 ctx_source_bias: mx.sym.Symbol,
                 cache: Optional[Dict[str, Optional[mx.sym.Symbol]]] = None) -> mx.sym.Symbol:
        # self-attention
        target_self_att = self.self_attention(inputs=self.pre_self_attention(target, None),
                                              bias=target_bias,
                                              cache=cache)
        target = self.post_self_attention(target_self_att, target)

        # target_aux = self.preattn_ff(self.pre_preattn_ff(target, None))
        # target_aux = self.post_preattn_ff(target_aux, target)


        # encoder attention
        target_enc_att = self.enc_attention(queries=self.pre_enc_attention(target, None),
                                            memory=source,
                                            bias=source_bias)
        target_main = self.post_enc_attention(target_enc_att, target)


        # ctx encoder attention
        target_ctx_enc_att = self.ctx_enc_attention(queries=self.pre_ctx_enc_attention(target_main, None),
                                                    memory=ctx_source,
                                                    bias=ctx_source_bias)
        target_ctx = self.post_ctx_enc_attention(target_ctx_enc_att, target_main)


        # target_main_ff = self.ff(self.pre_ff(target_main, None))
        # target_main = self.post_ff(target_main_ff, target_main)
        #
        # target_ctx_ff = self.ctx_ff(self.pre_ctx_ff(target_ctx, None))
        # target_ctx = self.post_ctx_ff(target_ctx_ff, target_ctx)

        target = self.gate(target_main, target_ctx, target)

        target_postgate_ff = self.postgate_ff(self.pre_postgate_ff(target, None))
        target = self.post_postgate_ff(target_postgate_ff, target)


        return target

class GateBlock:

    def __init__(self, num_hidden, dropout, act_type, prefix):
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act_type = act_type
        self.w_i2h = mx.sym.Variable('%si2h_weight' % prefix)
        self.b_i2h = mx.sym.Variable('%si2h_bias' % prefix)
        self.w_h2o = mx.sym.Variable('%sh2o_weight' % prefix)
        self.b_h2o = mx.sym.Variable('%sh2o_bias' % prefix)
        self.w_h2c = mx.sym.Variable('%sh2c_weight' % prefix)
        self.b_h2c = mx.sym.Variable('%sh2c_bias' % prefix)


        self.w_h2out = mx.sym.Variable('%sh2out_weight' % prefix)
        self.b_h2out = mx.sym.Variable('%sh2out_bias' % prefix)
        self.w_h2cut = mx.sym.Variable('%sh2cut_weight' % prefix)
        self.b_h2cut = mx.sym.Variable('%sh2cut_bias' % prefix)
        self.w_h2tout = mx.sym.Variable('%sh2tout_weight' % prefix)
        self.b_h2tout = mx.sym.Variable('%sh2tout_bias' % prefix)


        self.w_f2out = mx.sym.Variable('%sf2out_weight' % prefix)
        self.b_f2out = mx.sym.Variable('%sf2out_bias' % prefix)

    def __call__(self, target_main, target_ctx, target):

        z1 = mx.sym.FullyConnected(data=target_main, num_hidden=self.num_hidden, weight=self.w_i2h, bias=self.b_i2h, flatten=False)
        z2 = mx.sym.FullyConnected(data=target_ctx, num_hidden=self.num_hidden, weight=self.w_h2o, bias=self.b_h2o, flatten=False)

        z = z1 + z2
        z = layers.activation(z, act_type="sigmoid")

        # target_main = mx.sym.FullyConnected(data=target_main, num_hidden=self.num_hidden, weight=self.w_h2out, bias=self.b_h2out, flatten=False)
        # target_ctx = mx.sym.FullyConnected(data=target_ctx, num_hidden=self.num_hidden, weight=self.w_h2cut, bias=self.b_h2cut, flatten=False)

        target = z * (target_main) + (1 - z) * target_ctx

        return target

class GateTrgBlock:

    def __init__(self, num_hidden, dropout, act_type, prefix):
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act_type = act_type
        self.w_i2h = mx.sym.Variable('%si2h_weight' % prefix)
        self.b_i2h = mx.sym.Variable('%si2h_bias' % prefix)
        self.w_h2o = mx.sym.Variable('%sh2o_weight' % prefix)
        self.b_h2o = mx.sym.Variable('%sh2o_bias' % prefix)
        self.w_h2c = mx.sym.Variable('%sh2c_weight' % prefix)
        self.b_h2c = mx.sym.Variable('%sh2c_bias' % prefix)


        self.w_h2out = mx.sym.Variable('%sh2out_weight' % prefix)
        self.b_h2out = mx.sym.Variable('%sh2out_bias' % prefix)
        self.w_h2cut = mx.sym.Variable('%sh2cut_weight' % prefix)
        self.b_h2cut = mx.sym.Variable('%sh2cut_bias' % prefix)
        self.w_h2tout = mx.sym.Variable('%sh2tout_weight' % prefix)
        self.b_h2tout = mx.sym.Variable('%sh2tout_bias' % prefix)


        self.w_f2out = mx.sym.Variable('%sf2out_weight' % prefix)
        self.b_f2out = mx.sym.Variable('%sf2out_bias' % prefix)

    def __call__(self, target_main, target_ctx, target):

        z1 = mx.sym.FullyConnected(data=target_main, num_hidden=self.num_hidden, weight=self.w_i2h, bias=self.b_i2h, flatten=False)
        z2 = mx.sym.FullyConnected(data=target_ctx, num_hidden=self.num_hidden, weight=self.w_h2o, bias=self.b_h2o, flatten=False)
        z3 = mx.sym.FullyConnected(data=target, num_hidden=self.num_hidden, weight=self.w_h2c, bias=self.b_h2c, flatten=False)

        z = z1 + z2 + z3
        z = layers.activation(z, act_type="sigmoid")

        target = z * (target_main) + (1 - z) * target_ctx

        return target

class EncoderGateBlock:

    def __init__(self, num_hidden, dropout, act_type, prefix):
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act_type = act_type
        self.w_i2h = mx.sym.Variable('%si2h_weight' % prefix)
        self.b_i2h = mx.sym.Variable('%si2h_bias' % prefix)
        self.w_h2o = mx.sym.Variable('%sh2o_weight' % prefix)
        self.b_h2o = mx.sym.Variable('%sh2o_bias' % prefix)

        self.w_h2out = mx.sym.Variable('%sh2out_weight' % prefix)
        self.b_h2out = mx.sym.Variable('%sh2out_bias' % prefix)
        self.w_h2cut = mx.sym.Variable('%sh2cut_weight' % prefix)
        self.b_h2cut = mx.sym.Variable('%sh2cut_bias' % prefix)

        self.w_f2out = mx.sym.Variable('%sf2out_weight' % prefix)
        self.b_f2out = mx.sym.Variable('%sf2out_bias' % prefix)

    def __call__(self, source, ctx_source):

        z1 = mx.sym.FullyConnected(data=source, num_hidden=self.num_hidden, weight=self.w_i2h, bias=self.b_i2h, flatten=False)
        z2 = mx.sym.FullyConnected(data=ctx_source, num_hidden=self.num_hidden, weight=self.w_h2o, bias=self.b_h2o, flatten=False)

        z = z1 + z2
        z = layers.activation(z, act_type="sigmoid")

        source = z * source + (1 - z) * ctx_source

        return source

    def previous_call(self, source, ctx_source):

        z = mx.sym.FullyConnected(data=mx.sym.concat(source, ctx_source, dim=2), num_hidden=self.num_hidden, weight=self.w_i2h, bias=self.b_i2h, flatten=False)
        z = layers.activation(z, act_type="sigmoid")
        source = z * source + (1 - z) * ctx_source

        return source

class TransformerProcessBlock:
    """
    Block to perform pre/post processing on layer inputs.
    The processing steps are determined by the sequence argument, which can contain one of the three operations:
    n: layer normalization
    r: residual connection
    d: dropout
    """

    def __init__(self,
                 sequence: str,
                 num_hidden: int,
                 dropout: float,
                 prefix: str) -> None:
        self.sequence = sequence
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.prefix = prefix
        self.layer_norm = None
        if "n" in sequence:
            self.layer_norm = layers.LayerNormalization(num_hidden=self.num_hidden, prefix="%snorm" % self.prefix)

    def __call__(self,
                 data: mx.sym.Symbol,
                 prev: Optional[mx.sym.Symbol]) -> mx.sym.Symbol:
        """
        Apply processing sequence to data with optional previous input.

        :param data: Input data. Shape: (batch, length, num_hidden).
        :param prev: Previous data. Shape: (batch, length, num_hidden).
        :return: Processed data. Shape: (batch, length, num_hidden).
        """
        if not self.sequence:
            return data

        if prev is None:
            assert 'r' not in self.sequence, "Residual connection not allowed if no previous value given."

        for step in self.sequence:

            if step == "r":
                data = mx.sym._internal._plus(data, prev, name="%sresidual" % self.prefix)

            elif step == "n":
                data = self.layer_norm.normalize(data)

            elif step == "d":
                if self.dropout > 0.0:
                    data = mx.sym.Dropout(data, p=self.dropout, name="%sdropout" % self.prefix)
            else:
                raise ValueError("Unknown step in sequence: %s" % step)

        return data


class TransformerFeedForward:
    """
    Position-wise feed-forward network with activation.
    """

    def __init__(self,
                 num_hidden: int,
                 num_model: int,
                 act_type: str,
                 dropout: float,
                 prefix: str) -> None:
        self.num_hidden = num_hidden
        self.num_model = num_model
        self.dropout = dropout
        self.prefix = prefix
        self.act_type = act_type
        self.w_i2h = mx.sym.Variable('%si2h_weight' % prefix)
        self.b_i2h = mx.sym.Variable('%si2h_bias' % prefix)
        self.w_h2o = mx.sym.Variable('%sh2o_weight' % prefix)
        self.b_h2o = mx.sym.Variable('%sh2o_bias' % prefix)

    def __call__(self, x) -> mx.sym.Symbol:
        """
        Position-wise feed-forward network with activation.

        :param x: Symbol of shape (batch_size, seq_len, num_hidden)
        :return: Symbol of shape (batch_size, seq_len, num_hidden)
        """
        h = mx.sym.FullyConnected(data=x, num_hidden=self.num_hidden, weight=self.w_i2h, bias=self.b_i2h, flatten=False)
        h = layers.activation(h, act_type=self.act_type)
        if self.dropout > 0.0:
            h = mx.sym.Dropout(h, p=self.dropout)
        y = mx.sym.FullyConnected(data=h, num_hidden=self.num_model, weight=self.w_h2o, bias=self.b_h2o, flatten=False)
        return y


class VariableLengthBias(mx.operator.CustomOp):
    """
    Returns bias/mask given a vector of sequence lengths.
    """

    def __init__(self, max_length: int) -> None:
        super().__init__()
        self.max_length = max_length

    def forward(self, is_train, req, in_data, out_data, aux):
        # lengths: (batch_size,)
        lengths = in_data[0]

        # (max_length, batch_size)
        data = mx.nd.zeros((self.max_length, lengths.shape[0]), ctx=lengths.context)
        data = mx.nd.SequenceMask(data=data,
                                  use_sequence_length=True,
                                  sequence_length=lengths,
                                  value=C.LARGE_NEGATIVE_VALUE)
        # (batch_size, max_length)
        data = mx.nd.swapaxes(data, dim1=0, dim2=1)
        self.assign(out_data[0], req[0], data)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("variable_length_bias")
class VariableLengthBiasProp(mx.operator.CustomOpProp):

    def __init__(self, max_length: str) -> None:
        super().__init__()
        self.max_length = int(max_length)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        batch_size = in_shape[0][0]
        return in_shape, [(batch_size, self.max_length)], []

    def infer_type(self, in_type):
        return in_type, [np.float32], []

    def create_operator(self, ctx, shapes, dtypes):
        return VariableLengthBias(max_length=self.max_length)


def get_variable_length_bias(lengths: mx.sym.Symbol,
                             max_length: int,
                             num_heads: Optional[int] = None,
                             fold_heads: bool = True,
                             name: str = '') -> mx.sym.Symbol:
    """
    Returns bias/mask for variable sequence lengths.

    :param lengths: Sequence lengths. Shape: (batch,).
    :param max_length: Maximum sequence length.
    :param num_heads: Number of attention heads.
    :param fold_heads: Whether to fold heads dimension into batch dimension.
    :param name: Name of symbol.
    :return: Bias symbol.
    """
    # (batch_size, max_length)
    x = mx.symbol.Custom(data=lengths, max_length=max_length, op_type='variable_length_bias')
    if num_heads is not None:
        # (batch_size, heads, max_length) if fold_heads == False else (batch_size * heads, max_length)
        x = layers.broadcast_to_heads(x, num_heads, ndim=2, fold_heads=fold_heads)
    return mx.sym.BlockGrad(x, name='%sbias' % name)


def get_autoregressive_bias(max_length: int, name: str) -> mx.sym.Symbol:
    """
    Returns bias/mask to ensure position i can only attend to positions <i.

    :param max_length: Sequence length.
    :param name: Name of symbol.
    :return: Bias symbol of shape (1, max_length, max_length).
    """
    return mx.sym.BlockGrad(mx.symbol.Custom(length=max_length,
                                             name=name,
                                             op_type='auto_regressive_bias'))


class AutoRegressiveBias(mx.operator.CustomOp):
    """
    Returns a symbol of shape (1, length, length) with cells above the main diagonal
    set to a large negative value, e.g.
    length=4

    0 1 1 1
    0 0 1 1   * -99999
    0 0 0 1
    0 0 0 0
    """

    def __init__(self, length: int) -> None:
        super().__init__()
        self.bias = self.get_bias(length)

    @staticmethod
    def get_bias(length: int):
        # matrix with lower triangle and main diagonal set to 0, upper triangle set to 1
        upper_triangle = np.triu(np.ones((length, length)), k=1)
        # (1, length, length)
        bias = -99999999. * np.reshape(upper_triangle, (1, length, length))
        return mx.nd.array(bias)

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], self.bias)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("auto_regressive_bias")
class AutoRegressiveBiasProp(mx.operator.CustomOpProp):

    def __init__(self, length: str) -> None:
        super().__init__()
        self.length = int(length)

    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [], [(1, self.length, self.length)], []

    def infer_type(self, in_type):
        return [], [np.float32], []

    def create_operator(self, ctx, shapes, dtypes):
        return AutoRegressiveBias(length=self.length)
