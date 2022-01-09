# -*- coding: utf-8 -*-
# env: python 3
# Author: Kaiyu Huang, Wei Liu
# Copyright 2020 The DUTNLP Authors

import re
import sys
import os
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="demo of utils")
    parser.add_argument('-f', '--function', required=True, help='function list')
    parser.add_argument('-i', '--input', required=True, help='the path of input file')
    parser.add_argument('-ie', '--in_encode', required=False, help='the type of input file (defalut:utf8)')
    parser.add_argument('-o', '--output', required=True, help='the path of output file')
    parser.add_argument('-oe', '--out_encode', required=False, help='the type of output file (defalut:utf8)')
    parser.add_argument('-c', '--criterion', required=False, help='the criterion which you pointed')
    parser.add_argument('-io', '--input_ori', required=False, help='the path of original file at post processing step')    
    return parser

def add_tag(tag, input_file, output_file):
    # add a specific tag in the front of the sentences in corpus
    # all code needs to transfer to 'utf8' first
    # tag will be added a 'criterion<@@>sentence' for convenient
    tag = tag + '<@@>'
    with open(input_file, 'r', encoding='utf8') as f:
        data = f.read().splitlines()
    with open(output_file, 'w', encoding='utf8') as f:
        for line in data:
            f.write(tag + line + '\n')
    print('add <' + tag + '> in the ' + output_file + ' done!')

def tohalfwidth(data):
    new_data = list()
    for sentence in data:
        if sentence == '':
            continue
        new_sent = list()
        words = sentence.split()
        for word in words:
            half_word = ''
            for uchar in word:
                inside_code = ord(uchar)
                if inside_code == 12288:
                    inside_code = 32
                elif 65281 <= inside_code <= 65374:
                    inside_code -= 65248
                half_word += chr(inside_code)
            new_sent.append(half_word)
        new_data.append(new_sent)
    return new_data


def tohalfwidth_testline(data):
    new_data = list()
    for sentence in data:
        if sentence == '':
            continue
        new_sent = ''
        for uchar in sentence:
            inside_code = ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif 65281 <= inside_code <= 65374:
                inside_code -= 65248
            new_sent += chr(inside_code)
        new_data.append(new_sent)
    return new_data


def generalize(data):
    rNUM = re.compile(r'(-|\+)?(\d+)([\.|·/∶:]\d+)?%?')
    rENG = re.compile(r'[A-Za-z_.]+')
    #rRomaGreece = re.compile(r'[\u2160-\u216f|\u0370-\u03FF]+')
    new_data = list()
    for sent in data:
        new_sent = list()
        for word in sent:
            word = re.sub(r'\s+', '', word)
            word = re.sub(rNUM, '0', word)
            word = re.sub(rENG, 'X', word)
            #word = re.sub(rRomaGreece, 'X', word)
            #word = re.sub(rCon, '-', word)
            new_sent.append(word)
        new_data.append(new_sent)
    return new_data


def generalize_testline(data):
    rNUM = re.compile(r'(-|\+)?(\d+)([\.|·/∶:]\d+)?%?')
    rENG = re.compile(r'[A-Za-z_.]+')
    #rRomaGreece = re.compile(r'[\u2160-\u216f|\u0370-\u03FF]+')
    new_data = list()
    for sent in data:
        sent = re.sub(r'\s+', '', sent)
        sent = re.sub(rNUM, '0', sent)
        sent = re.sub(rENG, 'X', sent)
        #sent = re.sub(rRomaGreece, 'X', sent)
        new_data.append(sent)
    return new_data

def postprocess(origin_file, segment_file, output_file):
    with open(origin_file, 'r', encoding='UTF-8') as f_origin:
        data_origin = f_origin.readlines()
    with open(segment_file, 'r', encoding='UTF-8') as f_seg:
        data_seg = f_seg.readlines()

    # data_origin转半角
    data_half = list()
    for sentence in data_origin:
        newsent = list()
        for uchar in sentence:
            inside_code = ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif inside_code >= 65281 and inside_code <= 65374:
                inside_code -= 65248
            newchar = chr(inside_code)
            newsent.append(newchar)
        data_half.append(''.join(newsent))

    # 泛化字符恢复
    rNUM = re.compile(r'(-|\+)?(\d+)([\.|·/∶:]\d+)?(%)?')
    rENG = re.compile(r'[A-Za-z_.]+')
    index = 0
    data_general_recover = list()  # 泛化恢复后的分词结果
    for seg_line, origin_line in zip(data_seg, data_half):
        index += 1
        new_word_list = list()
        list_NUM = list()
        list_ENG = list()

        list_NUM_i = 0
        list_ENG_i = 0

        for t in re.findall(rNUM, origin_line):
            list_NUM.append(''.join(t))
        for t in re.findall(rENG, origin_line):
            list_ENG.append(''.join(t))
        word_list = seg_line.strip().split()
        for word in word_list:
            new_word = list()
            for i in range(len(word)):
                c = word[i]
                if c == '0':
                    new_word.append(list_NUM[list_NUM_i])
                    list_NUM_i += 1
                elif c == 'X':
                    #print(word_list)
                    new_word.append(list_ENG[list_ENG_i])
                    list_ENG_i += 1
                else:
                    new_word.append(c)
            new_word = ''.join(new_word)
            new_word_list.append(new_word)
        data_general_recover.append(new_word_list)

    # 切分原始的全角文件
    data_result = list()
    for seg_list, line_origin in zip(data_general_recover, data_origin):
        new_line = []
        len_list = []
        for item in seg_list:
            len_list.append(len(item))

        line_origin = line_origin.strip()
        i = 0
        for w_len in len_list:
            new_line.append(line_origin[i:i + w_len])
            i += w_len
        data_result.append(new_line)
    write(data_result, output_file) # gemerate utf8 file


def write(data, output_file, encode='utf8'):
    with open(output_file, 'w', encoding=encode) as f:
        for sent in data:
            if not sent:
                continue
            for word in sent:
                f.write(word + ' ')
            f.write('\n')
    print(output_file + ' write done!')

def write_testline(data, output_file, encode='utf8'):
    with open(output_file, 'w', encoding=encode) as f:
        for sent in data:
            if not sent:
                continue
            f.write(sent+'\n')
    print(output_file + ' write done!')

def preprocess(mode, input_file, output_file, encode='utf8'):
    # original seg data -> seg train data
    with open(input_file, 'r', encoding=encode) as f:
        data = f.read().splitlines()
    if mode == 'train':
        data = tohalfwidth(data)
        data = generalize(data)  # data generalize
        write(data, output_file) # gemerate utf8 file
    elif mode == 'test':
        data = tohalfwidth_testline(data)
        data = generalize_testline(data)  # data generalize
        write_testline(data, output_file) # generate utf8 file
    else:
        print('the mode is mistake (only support train and test now). pls check.')


if __name__ == '__main__':
    
    parser = create_parser()
    args = parser.parse_args()
    if args.function == 'p-train':
        if os.path.isfile(args.input):
            preprocess('train', args.input, args.output)
        else:
            print('the path of intput file is mistake. pls check.')
    elif args.function == 'p-test':
        if os.path.isfile(args.input):
            preprocess('test', args.input, args.output)
        else:
            print('the path of intput file is mistake. pls check.')
    elif args.function == 'add-tag':
        if args.criterion and os.path.isfile(args.input):
            add_tag(args.criterion, args.input, args.output)
        elif args.criterion:
            print('the path of intput file is mistake. pls check.')
        else:
            print('the criterion doesn\'t be pointed. pls check.')
    elif args.function == 'post':
        if os.path.isfile(args.input) and os.path.isfile(args.input_ori):
            postprocess(args.input_ori, args.input, args.output)
        else:
            print('the path of intput or original file is mistake. pls check.') 		
    print('preprocess done!')
