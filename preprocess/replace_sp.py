#!/usr/bin/python
#-*-coding:utf-8 -*-
# author: mld
# email: miradel51@126.com
# date : 2020/11/30
#time : 23:02 

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

origin_str = sys.argv[1]
out_str = sys.argv[2]

origin_str_f = open(origin_str,'r')
out_f = open(out_str,'w')

cont = ""

for eachline in origin_str_f:
	
	cont = eachline.decode('utf-8')
	cont = cont.strip().split("　")
			
	out_f.write(" ".join(cont))
	out_f.write('\n')
	

origin_str_f.close()
out_f.close()

