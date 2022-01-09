#!/usr/bin/python
#-*-coding:utf-8 -*-
# author: mld
# email: miradel51@126.com
# date : 2019/02/06
#time : 02:00(am)

import sys

origin_str = sys.argv[1]
out_str = sys.argv[2]

origin_str_f = open(origin_str,'r')
out_f = open(out_str,'w')

cont = ""
out_char = ""

for eachline in origin_str_f:
	
	#cont = eachline.decode('utf-8')
	cont = eachline.strip().split()
	
	for each_word in cont:	
		for each_chr in each_word:
			out_char += ' '+ each_chr
			
	out_f.write(out_char.strip())
	out_f.write('\n')
	
	#clear current out_char
	out_char = ""

origin_str_f.close()
out_f.close()

