from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert import BertPreTrainedModel, BertModel

class BertForTokenClassificationWithCRF(BertPreTrainedModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels=2):
        super(BertForTokenClassificationWithCRF, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels + 2)
        self.apply(self.init_bert_weights)
        
        self.START_TAG_IDX, self.END_TAG_IDX = -2, -1
        init_transitions = torch.randn(num_labels + 2, num_labels + 2)
        init_transitions[:, self.START_TAG_IDX] = -10000.
        init_transitions[self.END_TAG_IDX, :] = -10000.
        self.transitions = nn.Parameter(init_transitions)

    def log_sum_exp(self, vec, m_size):
        _, idx = torch.max(vec, 1)
        max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)
        return max_score.view(-1, m_size) + torch.log(torch.sum(
                torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)
    
    def _forward_alg(self, feats, mask=None, mrt=False):
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)

        mask = mask.transpose(1, 0).contiguous()
        ins_num = batch_size * seq_len
        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = self.log_sum_exp(cur_values, tag_size)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            if masked_cur_partition.dim() != 0:
                mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
                partition.masked_scatter_(mask_idx, masked_cur_partition)
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size) + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = self.log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, self.END_TAG_IDX]
        return final_partition, scores
    
    def _score_sentence(self, scores, mask, tags, mrt=False):
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(-1)
        
        new_tags = tags.clone()
        new_tags[ : , 0] = tags[ : , 0] + (tag_size - 2) * tag_size
        new_tags[ : , 1 : ] = tags[: , 1 : ] + tags[ : , : -1] * tag_size

        end_transition = self.transitions[:, self.END_TAG_IDX].contiguous().view(
            1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        
        end_ids = torch.gather(tags, 1, length_mask-1)
        end_energy = torch.gather(end_transition, 1, end_ids)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(
            seq_len, batch_size)

        tg_energy = tg_energy * mask.transpose(1, 0).float()
        gold_score = tg_energy.sum(dim=0) + end_energy.view(-1)
        return gold_score
    
    def neg_log_likelihood_loss(self, feats, mask, tags, mrt=False):
        batch_size = feats.size(0)
        forward_score, scores = self._forward_alg(feats, mask, mrt=mrt)
        gold_score = self._score_sentence(scores, mask, tags, mrt=mrt)
        if mrt:
            return forward_score - gold_score
        else:
            return (forward_score - gold_score).sum() / batch_size
        
    def _viterbi_decode(self, feats, mask=None):
        """
        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)

        Returns:
            decode_idx: (batch_size, seq_len), viterbi decode results
            path_score: size=(batch_size, 1), score of each sentence
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        __feats = feats.clone()
        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter = enumerate(scores)
        # record the position of the best score
        back_points = list()
        partition_history = list()
        mask = (1 - mask.long()).byte()
        _, inivalues = seq_iter.__next__()
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        partition_history.append(partition)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition.unsqueeze(-1))

            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)

        partition_history = torch.cat(partition_history).view(
            seq_len, batch_size, -1).transpose(1, 0).contiguous()

        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(
            partition_history, 1, last_position).view(batch_size, tag_size, 1)

        last_values = last_partition.expand(batch_size, tag_size, tag_size) +             self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = (__feats[ : , 0, : ] * 0.0).long()
        
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        pointer = last_bp[:, self.END_TAG_IDX]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()

        back_points.scatter_(1, last_position, insert_last)

        back_points = back_points.transpose(1, 0).contiguous()

        decode_idx = mask.long() * 0
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.view(-1).data
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, mrt=False):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        mask = (input_ids != 0)[ : , 2 : ]
        logits = logits[ : , 1 : -1]

        if labels is not None:
            if mrt:
                batch_size = input_ids.shape[0]
                n_mrt = labels.shape[0] // input_ids.shape[0]
                seql = mask.shape[1]
                logits = logits.unsqueeze(1).expand(batch_size, n_mrt, seql, -1).reshape(batch_size * n_mrt, seql, -1)
                mask = mask.unsqueeze(1).expand(batch_size, n_mrt, seql).reshape(batch_size * n_mrt, seql) 
                loss = self.neg_log_likelihood_loss(logits, mask, labels, mrt=True).reshape(batch_size, n_mrt)
            else:
                loss = self.neg_log_likelihood_loss(logits, mask, labels, mrt=False)
            return loss
        else:
            path_score, tag_space = self._viterbi_decode(logits, mask)
            return tag_space