"""
@Time   :   2025-03-24 20:24:00
@File   :   modeling_soft_masked_bert_onnx.py
@Author :   Abtion, Zhang Chen
@Email  :   abtion{at}outlook.com, zhangchen.shaanxi{at}gmail.com
"""
import operator
import os
from collections import OrderedDict
import transformers as tfs
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertOnlyMLMHead
from transformers.modeling_utils import ModuleUtilsMixin
from bbcm.engine.csc_trainer import CscTrainingModel
import numpy as np


class DetectionNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            self.config.hidden_size,
            self.config.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=self.config.hidden_dropout_prob,
            bidirectional=True,
        )
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(self.config.hidden_size, 1)

    def forward(self, hidden_states):
        out, _ = self.gru(hidden_states)
        prob = self.linear(out)
        prob = self.sigmoid(prob)
        return prob


class BertCorrectionModel(torch.nn.Module, ModuleUtilsMixin):
    def __init__(self, config, tokenizer, device):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embeddings = BertEmbeddings(self.config)
        self.corrector = BertEncoder(self.config)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.cls = BertOnlyMLMHead(self.config)
        self._device = device

    def forward(self, prob, embed=None, attention_mask=None,residual_connection=False):


        # 此处较原文有一定改动，做此改动意在完整保留type_ids及position_ids的embedding。
        mask_embed = self.embeddings(torch.ones_like(prob.squeeze(-1)).long() * self.mask_token_id).detach()
        # 此处为原文实现
        # mask_embed = self.embeddings(torch.tensor([[self.mask_token_id]], device=self._device)).detach()
        cor_embed = prob * mask_embed + (1 - prob) * embed

        input_shape = embed.size()
        device = embed.device

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask,
                                                                                 input_shape, device)
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        # print(f"cor_embed.shape: {cor_embed.shape}")
        encoder_outputs = self.corrector(
            cor_embed,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=False,
        )
        sequence_output = encoder_outputs[0]

        sequence_output = sequence_output + embed if residual_connection else sequence_output
        prediction_scores = self.cls(sequence_output)
        out = (prediction_scores, sequence_output)


        return out

    def load_from_transformers_state_dict(self, gen_fp):
        state_dict = OrderedDict()
        gen_state_dict = tfs.AutoModelForMaskedLM.from_pretrained(gen_fp).state_dict()
        for k, v in gen_state_dict.items():
            name = k
            if name.startswith('bert'):
                name = name[5:]
            if name.startswith('encoder'):
                name = f'corrector.{name[8:]}'
            if 'gamma' in name:
                name = name.replace('gamma', 'weight')
            if 'beta' in name:
                name = name.replace('beta', 'bias')
            state_dict[name] = v
        self.load_state_dict(state_dict, strict=False)


class SoftMaskedBertModel(CscTrainingModel):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.config = tfs.AutoConfig.from_pretrained(cfg.MODEL.BERT_CKPT)
        self.detector = DetectionNetwork(self.config)
        self.tokenizer = tokenizer
        self.corrector = BertCorrectionModel(self.config, tokenizer, cfg.MODEL.DEVICE)
        self.corrector.load_from_transformers_state_dict(self.cfg.MODEL.BERT_CKPT)
        self._device = cfg.MODEL.DEVICE

    def forward(self, input_ids, attention_mask, token_type_ids):
        # print(f"texts: {len(texts)}")
        # [print((x,len(x))) for x in texts]
        # encoded_texts = self.tokenizer(texts, padding=True, return_tensors='pt', truncation=True,)
        # encoded_texts.to(self._device)
        # print(f"encoded_texts['input_ids'].shape: {encoded_texts['input_ids'].shape}")
        embed = self.corrector.embeddings(input_ids=input_ids,
                                          token_type_ids=token_type_ids,
                                          )
        # print(f"embed.shape: {embed.shape}")
        prob = self.detector(embed)
        # print()
        # print(f"prob.shape: {prob.shape}")
        # print(f"cor_labels.shape: {(len(cor_labels),len(cor_labels[0]),'...')}")
        # [print((x,len(x))) for x in cor_labels]
        # cor_out = self.corrector(texts, prob, embed, cor_labels,
        #                          residual_connection=True)
        cor_out = self.corrector(prob, embed,attention_mask,
                                 residual_connection=True)


        outputs = (prob.squeeze(-1),) + cor_out

        return outputs

    def load_from_transformers_state_dict(self, gen_fp):
        """
        从transformers加载预训练权重
        :param gen_fp:
        :return:
        """
        self.corrector.load_from_transformers_state_dict(gen_fp)
