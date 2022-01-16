# -*- coding: utf-8 -*-
# env: python 3
# Author: Kaiyu Huang, Wei Liu
# Copyright 2020 The DUTNLP Authors

import argparse
import re
import random
import string
import os
import copy
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForTokenClassification
from torch.nn.utils.rnn import pad_sequence
from pytorch_pretrained_bert.optimization import BertAdam
import torch.nn as nn
import torch.nn.functional as F

from crfmodel import BertForTokenClassificationWithCRF

def create_parser():
    parser = argparse.ArgumentParser(description="demo of utils")
    parser.add_argument('-c', '--config', required=True, help='the path of configure file')
    return parser

class Dataset(Dataset):
    def __init__(self, x_, y_):
        self.x_data = x_
        self.y_data = y_
        self.len = len(x_)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class MRTDataset(Dataset):
    def __init__(self, x_, y_, r_, config):
        assert(len(x_) % config['n_mrt'] == 0)
        
        self.len = len(x_) // config['n_mrt']
        self.x_data = []
        self.y_data = []
        self.r_data = []
        
        for i in range(0, self.len):
            start = i * config['n_mrt']
            end = (i + 1) * config['n_mrt']
            max_len = -100000000
            min_len = 100000000
            for j in range(start, end):
                max_len = max(len(x_[j]), max_len)
                min_len = min(len(x_[j]), min_len)
            assert(max_len == min_len)
            self.x_data.append(x_[start])
            self.y_data.append(y_[start : end])
            self.r_data.append(r_[start : end])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.r_data[index]

    def __len__(self):
        return self.len


def collate_fn(data):
    inputs = list()
    labels = list()
    seq_len = list()
    for key in data:
        x, y = key
        inputs.append(x)
        labels.append(y)
        seq_len.append(len(y))
    datax = pad_sequence([torch.from_numpy(np.array(x)) for x in inputs], batch_first=True, padding_value=0).long()
    datay = pad_sequence([torch.from_numpy(np.array(y)) for y in labels], batch_first=True, padding_value=0).long()
    data =  [datax, datay, seq_len]
    return data


def mrt_collate_fn(data):
    inputs = list()
    labels = list()
    seq_len = list()
    risks = list()
    for x, y, r in data:
        inputs.append(x)
        seq_len.append(len(y[0]))
        for __y, __r in zip(y, r):
            labels.append(__y)
            risks.append(__r)
    datax = pad_sequence([torch.from_numpy(np.array(x)) for x in inputs], batch_first=True, padding_value=0).long()
    datay = pad_sequence([torch.from_numpy(np.array(y)) for y in labels], batch_first=True, padding_value=0).long()
    data =  [datax, datay, seq_len, risks]
    return data


def word2label(words):
    labels = []
    for word in words:
        if len(word) == 1:
            labels.append(4)
        elif len(word) == 2:
            labels.append(1)
            labels.append(2)
        else:
            labels.append(1)
            n2 = len(word)-2
            labels = labels + [3] * n2
            labels.append(2)
    return labels


def get_single_data(data, __risks, config):
    sents = list()
    labels = list()
    risks = list()
    for sent, r in zip(data, __risks):
        Left = 0
        sent = sent.split()
        if len(sent) == 0:
            continue
        curr_len = 0
        __sent = []
        for word in sent:
            if curr_len < config['maxlen']:
                if curr_len + len(word) <= config['maxlen']:
                    __sent.append(word)
                else:
                    __sent.append(word[ : config['maxlen'] - curr_len])
                    assert(len(''.join(__sent)) == config['maxlen'])
                curr_len += len(word)
            else:
                break
        sents.append(list(''.join(__sent)))
        labels.append(word2label(__sent))
        risks.append(r)
    return sents, labels, risks


def split_single_data(data, config):
    sents = list()
    labels = list()
    for sent in data:
        Left = 0
        sent = list(sent)
        for idx,c in enumerate(sent):
            if c not in config['special_token']:
                if len(re.sub('\W','',c,flags=re.U))==0:
                    if idx > Left:
                        sents.append(list(''.join(sent[Left:idx])))
                        labels.append([0]*len(list(''.join(sent[Left:idx]))))
                        sents.append(c)
                    else:
                        sents.append(c)
                    Left = idx+1
        if Left!=len(sent):
            sents.append(list(''.join(sent[Left:])))
            labels.append([0]*len(list(''.join(sent[Left:]))))
        sents.append('\n')
    return sents, labels


def prepare_data(config):
    # prepare the train/test data for the model
    print('Loading data......')
    with open(config['load_file'], 'r', encoding='utf8') as f:
        data = f.read().splitlines()
    if config['mode'] == 'train':
        with open(config['risk_file'], 'r') as f:
            __risks = [float(x.strip()) for x in f]
        sents, labels, risks = get_single_data(data, __risks, config)
        assert(len(sents) == len(risks))
        assert(len(labels) == len(risks))
    elif config['mode'] == 'test':
        sents, labels = split_single_data(data, config)
    elif config['mode'] == 'valid':
        with open(config['valid_file'], 'r', encoding='utf8') as f:
            data = f.read().splitlines()
        sents, labels = split_single_data(data, config)

    print('Load data done!')
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    vocab = tokenizer.vocab
    config['vocab_size'] = len(vocab)
    idx = list()
    for sent in sents:
        if type(sent) != list:
            continue
        tokenized_text = copy.deepcopy(sent)
        for i, c in enumerate(tokenized_text):
            if c in config['token_mapping_rule']:
                tokenized_text[i] = config['token_mapping_rule'][c]
            elif c not in vocab:
                tokenized_text[i] = '[UNK]'
        tokenized_text.insert(0,'[CLS]')
        tokenized_text.append('[SEP]')
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        idx.append(indexed_tokens)
    if config['mode'] == 'train':
        train_data = MRTDataset(idx, labels, risks, config)
        train_data_loader = DataLoader(dataset=train_data, batch_size=config['batch_size'], shuffle=config['shuffle'], collate_fn=mrt_collate_fn)
        return train_data_loader
    elif config['mode'] == 'test' or config['mode'] == 'valid':
        test_data = Dataset(idx, labels)
        test_data_loader = DataLoader(dataset=test_data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
        return sents, test_data_loader


def predict(model, data_loader, config):
    result_matrix_list = []
    for batch, data in enumerate(data_loader):
        inputs, labels, seq_len = data
        segments_tensors = torch.zeros(inputs.size(0), inputs.size(1), dtype=torch.int64)
        logits = model(input_ids=inputs.to(config['device']), token_type_ids=segments_tensors.to(config['device']), attention_mask=None, labels=None)
        
        if config['CRF']:
            tag_space = logits
        else:
            tag_space = F.softmax(logits, dim=2)
            tag_space = tag_space.argmax(dim=2)

        result_matrix = list(tag_space.cpu().numpy())
        if config['CRF']:
            result_matrix = [result_matrix[i][ : eof] for i, eof in enumerate(seq_len)]
        else:
            result_matrix = [result_matrix[i][1:eof+1] for i, eof in enumerate(seq_len)]
        
        result_matrix_list += result_matrix
        print("\rThe process is in %d of %d ! " % (batch+1, len(data_loader)), end='')
    return result_matrix_list


def restore_result(sents, results):
    new_results = list()
    i = 0
    j = 0
    word_result = list()
    while i < len(sents):
        if type(sents[i]) != list:
            if sents[i] == '\n':
                new_results.append(word_result)
                word_result = list()
            else:
                word_result.append(sents[i])
            i += 1
        elif len(sents[i]) != len(results[j]):
            s = ''
            for c in sents[i]:
                s = s + c
            print('The result with ' + str(len(results[j])) + ' lengths does not match with the sent of ' + str(s) + ' with ' + str(len(sents[i])))
            sys.exit(1)
        else:
            word = ''
            for k in range(len(sents[i])):
                if results[j][k] == 4:
                    if word != '':
                        word_result.append(word)
                    word_result.append(sents[i][k])
                    word = ''
                elif results[j][k] == 1:
                    if word != '':
                        word_result.append(word)
                        word = ''
                    word = sents[i][k]
                elif results[j][k] == 2:
                    word += sents[i][k]
                    word_result.append(word)
                    word = ''
                elif results[j][k] == 3:
                    word += sents[i][k]
                else:
                    if word != '':
                        word_result.append(word)
                    word_result.append(sents[i][k])
                    word = ''
            if word != '':
                word_result.append(word)
            i += 1
            j += 1
    if len(word_result) != 0:
        new_results.append(word_result)
    return new_results

def write(data, filename):
    with open(filename, 'w', encoding='utf8') as f:
        for sent in data:
            for word in sent:
                f.write(word + ' ')
            f.write('\n')
    print ('the file has been writen successfully! ')

def score_shell(word_dict, gold_file, result_file, output_file):
    cmd = 'perl score ' + str(word_dict) + ' ' + str(gold_file) + ' ' + str(result_file) + '>' + str(output_file)
    res = os.system(cmd)
    return res


def train(config):
    print('-' * 50)
    print('Loading pre-trained transfer model......')
    
    if config['CRF']:
        model = BertForTokenClassificationWithCRF.from_pretrained(config['model_name'],num_labels=config['tagset_size']+1)
    else:
        model = BertForTokenClassification.from_pretrained(config['model_name'],num_labels=config['tagset_size']+1)

    if config['update_model']:
        load_model_name = config['update_model']
        print('load update model name is : ' + load_model_name)
        checkpoint = torch.load(load_model_name)
        model.load_state_dict(checkpoint['net'])
    model.to(config['device'])
    print('Load pre-trained transfer model done!')
    print('-' * 50)
    print('Deploying the training data......')
    train_data_loader = prepare_data(config)
    len_train_data_loader = len(train_data_loader)
    print('The training data is %d batches with %d batch sizes' % (len(train_data_loader), config['batch_size']))
    if os.path.isfile(config['valid_file']):
        print('Dev process set ! Deploying the Dev data......')
        config['mode'] = config['mode'].replace('train','valid')
        valid_sents, valid_data_loader = prepare_data(config)
        print('The validation data has loaded done!')
        config['mode'] = config['mode'].replace('valid','train')
    else:
        print('the valid file ' + config['valid_file'] + ' is not exist pls check it in the config.txt')
    print('-' * 50)
    print('Train step! The model runs on ' + str(config['device']))
    loss_list = dict()
    # train set
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params':
                [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
                0.01
        },
        {
            'params':
                [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=config['lr'], schedule=None)
    loss_function = nn.NLLLoss(ignore_index=0, reduce=False)
    best_f1 = 0.0
    for epoch in range(config['epochs']):
        model.train()
        model.zero_grad()
        total_loss = 0
        batch_step = 0
        for batch, data in enumerate(train_data_loader):
            batch_step += 1
            inputs, labels, seq_len, risks = data
            batch_size = inputs.size(0)
            seql = inputs.size(1) - 2
            segments_tensors = torch.zeros(inputs.size(0), inputs.size(1), dtype=torch.int64)
            labels = labels.to(config['device'])
            n_mrt = config['n_mrt']
            
            if config['CRF']:
                __loss = model(input_ids=inputs.to(config['device']), token_type_ids=segments_tensors.to(config['device']),
                                          attention_mask=None, labels=labels, mrt=True)
            else:
                logits = model(input_ids=inputs.to(config['device']), token_type_ids=segments_tensors.to(config['device']),
                                          attention_mask=None, labels=None)
                logits = logits[:, 1:-1, :]
                
                logits = logits.unsqueeze(1).expand(batch_size, n_mrt, seql, -1).reshape(batch_size * n_mrt * seql, -1)
                logits = F.log_softmax(logits, 1)
                
                __loss = loss_function(logits, labels.reshape(batch_size * n_mrt * seql)).reshape(batch_size, n_mrt, seql)
                seq_len = torch.tensor(seq_len).to(config['device']).float().unsqueeze(-1)
                __loss = torch.sum(__loss, dim=-1) / seq_len

            prob = F.softmax(__loss * -config['alpha_mrt'])
            reg_loss = torch.mean(-torch.sum(torch.exp(__loss * -config['alpha_mrt']), dim=-1))
            
            risks = torch.tensor(risks).to(config['device']).reshape(batch_size, n_mrt)
            mrt_loss = torch.mean(torch.sum(prob * risks, dim=1))
            loss = mrt_loss + config['reg_lambda'] * reg_loss
            
            total_loss += float(loss)
            loss.backward()
            
            optimizer.step()
            model.zero_grad()
            print("\rEpoch: %d ! the process is in %d of %d ! " % (epoch+1, batch+1, len(train_data_loader)),
                  end='')
        loss_avg = total_loss / batch_step
        loss_list[epoch] = loss_avg
        print("The loss is %f ! " % (loss_avg))

        # valid process
        if os.path.isfile(config['valid_file']) and os.path.isfile(config['gold_file']):
            model.eval()
            with torch.no_grad():
                valid_results = predict(model, valid_data_loader, config)
                valid_results = restore_result(valid_sents, valid_results)
                tmp_filename = ''.join(random.sample(string.ascii_letters + string.digits, 8))
                write(valid_results, tmp_filename+'.txt')
                # create an empty file
                ftmp = open(tmp_filename+'_dict.txt', 'w', encoding='utf8')
                ftmp.close()
                res = score_shell(tmp_filename+'_dict.txt', config['gold_file'], tmp_filename+'.txt', tmp_filename+'_score.txt')
                if res == 0:
                    get_score_cmd = 'grep \'F MEASURE\' ' + tmp_filename + '_score.txt'
                    f1 = os.popen(get_score_cmd).read().replace('\n', '')
                    print('The evaluation of epoch {} is {} !'.format(str(epoch+1),f1))
                else:
                    print('The command of score failed, pls check or remove the validation step')
                os.system('rm ' + tmp_filename+'_dict.txt')
                os.system('rm ' + tmp_filename+'.txt')
                os.system('rm ' + tmp_filename+'_score.txt')
        # model save process
        if config['model_path'] and (epoch+1) % int(config['save_model_epochs']) == 0:
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            if config['save_model_name']:
                model_name = os.path.join(config['model_path'], config['save_model_name']+'_'+str(epoch+1) + '.pkl')
            else:
                model_name = os.path.join(config['model_path'], str(epoch+1) + '.pkl')
            torch.save(state, model_name)
            print('The epoch %d is saved successfully, named %s !' % (epoch+1, model_name))

    print('train done!')


def test(config):
    print('-' * 50)
    print('Loading core model......')
    load_model_name = config['test_model']
    if os.path.exists(config['test_model']):
        pass
    else:
        print('the test model ' + config['test_model'] + ' is not exist pls check it in the config.txt')
    print('Core model name is : ' + load_model_name)

    if config['CRF']:
        model = BertForTokenClassificationWithCRF.from_pretrained(config['model_name'],num_labels=config['tagset_size']+1)
    else:
        model = BertForTokenClassification.from_pretrained(config['model_name'],num_labels=config['tagset_size']+1)

    print('-' * 50)
    print('Deploying the test data......')
    config['shuffle'] = False
    test_sents, test_data_loader = prepare_data(config)
    print('Test data loaded done!')
    model.to(config['device'])
    checkpoint = torch.load(load_model_name)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    print('-' * 50)
    print('core model loaded done! Start predicting......')
    with torch.no_grad():
        results = predict(model, test_data_loader, config)
        results = restore_result(test_sents, results)
        if config['output_file']:
            write(results, config['output_file'])
        else:
            print('The output file is not pointed. pls check the config.txt')
            sys.exit(1)
    print('Test process done!')


def load_config(config_data):
    new_config = dict()
    multi_line_flag = 0
    multi_key = ''
    multi_value = ''
    for token in config_data:
        config_list = token.split('=')
        if re.sub('\W', '', token) == '':
            continue
        if token[0] == '#':
            #it is a note
            continue
        elif len(config_list) != 2 and multi_line_flag == 0:
            #it is not a note but an error of configuration
            continue
        elif multi_line_flag == 1 and len(config_list) == 1:
            multi_value = multi_value + config_list[0].strip().replace('\'', '')
        elif multi_line_flag == 1 and len(config_list) == 2:
            new_config[multi_key] = multi_value
            multi_key = ''
            multi_value = ''
            multi_line_flag = 0
            key = config_list[0].rstrip()
            value = config_list[1].strip().replace('\'', '')
            if key == 'token_mapping_rule' and value[0] == '{':
                multi_line_flag = 1
                multi_key = key
                multi_value = value
            else:
                new_config[key] = value
        else:
            key = config_list[0].rstrip()
            value = config_list[1].strip().replace('\'', '')
            if key == 'token_mapping_rule' and value[0] == '{':
                multi_line_flag = 1
                multi_key = key
                multi_value = value
            else:
                new_config[key] = value
    return new_config


def update_config(config, config_data):
    for key in config_data:
        # check legitimacy and update
        if key == 'token_mapping_rule':
            token_mapping_rule = dict()
            if config_data[key][0] == '{' and config_data[key][-1] == '}':
                config_data[key] = config_data[key][1:-1]
                map_list = config_data[key].split(',')
                for token in map_list:
                    map_key = token.split(':')[0].strip()
                    map_value = token.split(':')[1].strip()
                    map_value_num = map_value.replace('[unused','').replace(']','')
                    try:
                        map_value_num = int(map_value_num)
                    except ValueError:
                        print('the format of token_mapping_rule token is not like [unused+NUM]')
                        sys.exit(1)
                    if 1 <= map_value_num <= 50:
                        token_mapping_rule[map_key] = map_value
                    else:
                        print('the value of token_mapping_rule token is out of scope (1-50)')
                        sys.exit(1)
            else:
                print('token_mapping_rule is not a dict pls check it in the config.txt')
                pass
            config[key] = token_mapping_rule
        elif key == 'mode':
            config_data[key] = config_data[key].lower()
            if config_data[key] == 'test' or config_data[key] == 'train':
                config[key] = config_data[key]
            else:
                print('the set of ' + key + ' is not exist pls check it in the config.txt')
                sys.exit(1)
        elif key == 'seed' or key == 'tagset_size' or key == 'batch_size' or key == 'save_model_epochs' or\
            key == 'maxlen' or key == 'epochs' or key == 'n_mrt':
            if int(config_data[key]) >= 0:
                config[key] = int(config_data[key])
            else:
                # negative number error
                pass
        elif key == 'model_name':
            if os.path.exists(config_data[key]):
                config[key] = config_data[key]
            else:
                print('the set of ' + key + ' points to the dir ' + config_data[key] + ' is not exist pls check it in the config.txt')
                sys.exit(1)
        elif key == 'model_path':
            if config_data[key] == '':
                pass
            elif os.path.exists(config_data[key]):
                config[key] = config_data[key]
            else:
                config[key] = config_data[key]
                os.makedirs(config_data[key])
                print('the set of ' + key + ' points to the dir ' + config_data[key] + ' is not exist. Creating the dir!')
        elif key == 'save_model_name':
            if config_data[key]:
                config[key] = re.sub('\W', '', config_data[key])
        elif key == 'output_file':
            if config_data[key]:
                config[key] = config_data[key]
            else:
                #non-output
                pass
        elif key == 'update_model' or key == 'special_token' or key == 'load_file' or key == 'risk_file':
            if os.path.exists(config_data[key]):
                if key == 'special_token':
                    with open(config_data[key], 'r', encoding='utf8') as f:
                        data = f.read().splitlines()
                    config[key] = data
                else:
                    config[key] = config_data[key]
            elif config_data[key] == '':
                # non-set of 'update_model' or 'special_token' or 'load_file' or 'valid_file' or 'test_model'
                pass
            else:
                print('the set of ' + key + ' points to the file ' + config_data[key] + ' is not exist pls check it in the config.txt')
                sys.exit(1)
        elif key == 'valid_file' or key == 'test_model' or key == 'gold_file':
            if config_data[key]:
                config[key] = config_data[key]
        elif key in ['shuffle', 'CRF']:
            config_data[key] = config_data[key].lower()
            if config_data[key] == 'true':
                config[key] = True
            elif config_data[key] == 'false':
                config[key] = False
            else:
                # illegal input
                pass
        elif key == 'CUDA_ENV_NUM':
            # check the cuda whether available
            config[key] = int(config_data[key])
        elif key == 'learning_rate':
            config['lr'] = float(config_data[key])
        elif key == 'alpha_mrt' or key == 'reg_lambda':
            config[key] = float(config_data[key])
        else:
            # illegal input
            pass
    return config


def main(config_file):
    print('initialize the configuration')
    token_mapping_rule = {'<pku>': '[unused4]', '/<pku>': '[unused5]',
                          '<zx>': '[unused6]', '/<zx>': '[unused7]',
                          '<msr>': '[unused8]', '/<msr>': '[unused9]',
                          '<ctb>': '[unused10]', '/<ctb>': '[unused11]',
                          '<sxu>': '[unused12]', '/<sxu>': '[unused13]',
                          '<udc>': '[unused14]', '/<udc>': '[unused15]',
                          '<cnc>': '[unused16]', '/<cnc>': '[unused17]',
                          'X': '[unused1]', '0': '[unused2]', 'M': '[unused3]'}
    config = {'mode':'', 'seed':812, 'model_name':'', 'tagset_size':4, 'update_model':'', 'epochs':1,
              'batch_size':1, 'model_path':'', 'save_model_epochs':1, 'save_model_name':'', 'valid_file':'', 'gold_file':'',  
              'load_file':'', 'risk_file':'', 'token_mapping_rule':token_mapping_rule, 'shuffle':True, 'maxlen':60, 'lr':2e-05, 
              'special_token':[], 'CUDA_ENV_NUM':0, 'test_model':'', 'output_file':'', 'n_mrt': 1, 'alpha_mrt': 0.5, 'reg_lambda': 0.0, 'CRF': False, }
    with open(config_file, 'r', encoding='utf8') as f:
        config_data = f.read().splitlines()
    # update config information
    config_data = load_config(config_data)
    config = update_config(config, config_data)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.manual_seed(config['seed'])
        device = torch.device('cuda')
        torch.cuda.set_device(int(config['CUDA_ENV_NUM']))
    else:
        torch.manual_seed(config['seed'])
        device = torch.device('cpu')
    config['device'] = device
    print(config)
    if config['mode'] == 'train':
        train(config)
    elif config['mode'] == 'test':
        test(config)


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    if os.path.isfile(args.config):
        main(args.config)
    else:
        print('the path of config file is mistake. pls check.')
    print('process done!')
