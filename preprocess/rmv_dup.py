#!/usr/bin/python
#_*_coding:utf-8_*_
#author: miradeljan
#email:miradel512126.com
#data:20201216
#time: 21:12

import sys
import datetime

if __name__ == '__main__':
	
	#flush buffer
	sys.stdout.flush()
	tb = datetime.datetime.now()

	print("Removing ...")

	ori_ = sys.argv[1]
	ret_ = sys.argv[2]

	ori_f = open(ori_,"r")
	ret_f = open(ret_,"w")

	ori_con = ""
	ret_con = {}

	senln, _dup = 0, 0

	for el in ori_f:

		senln += 1
		ori_con = el.strip()
		#old method
		#ret_con.update({ori_con: ""})
		#new method
		if ori_con in ret_con.keys():
			
			ret_con[ori_con] += 1
		else:
			ret_con[ori_con] = 1
	
	# calculate dup lines
	_dup = senln - len(ret_con.items())
	
	ori_f.close()
	ori_f = open(ori_,"r")	
	#read each items of dict and write new file
	if _dup == 0:
		#do not need to write new items of the dict, it should be same as original file
		for el in ori_f:
			ret_f.write(el.strip())
			ret_f.write("\n")
	else:

		for k in ret_con.keys():
			ret_f.write(k)
			ret_f.write("\n")


# show training time
te = datetime.datetime.now()
print('The removing task was started at',tb.strftime("%Y-%m-%d %H:%M:%S"))
print('Ended at',te.strftime("%Y-%m-%d %H:%M:%S"))
print('The total removing time is',format((te - tb).seconds / 60.0, '.2f'),'minutes')

ori_f.close()
ret_f.close()

print("*******************************************************")
print("There are",format(senln), 'lines in original file')
print("The total duplicated lines are:",format(_dup))
