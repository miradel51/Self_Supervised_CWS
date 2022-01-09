#!/usr/bin/python
#-*-coding:utf-8 -*-
# author: mld
# email: miradel51@126.com
# date : 2020/11/23
# time : 08:20(am)

import sys
import random

char_ = sys.argv[1]
class_= sys.argv[2]

char_f = open(char_,"r")
class_f =  open(class_,"w")

cont,gen_ = "",""
sln = 0

lb_=['b','m','e','s']

for echl in char_f:
	
	sln += 1
	cont = echl.strip().split()
	
	for _ in range(len(cont)):
		
		gen_ += lb_[random.randint(0,3)]+ " "

	gen_ = gen_.strip()
	# check the length
	if len(cont) != len(gen_.strip().split()):
		print("Please check the line {} in input file".format(sln))
		break
	else:
		class_f.write(gen_)
		class_f.write("\n")
	
	#clear generated random rabels
	gen_ = ""

char_f.close()
class_f.close()
		


	
