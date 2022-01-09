#!/usr/bin/python
#-*-coding:utf-8 -*-
# author: mld
# email: miradel51@126.com
# date : 2020/12/02
# time : 10:12(am)

import sys
import pdb
import time

class Timer(object):

	def __init__(self):
		self.total = 0

	def start(self):
		self.start_time = time.time()
		return self.start_time

	def finish(self):
		self.total += time.time() - self.start_time
		return self.total


def get_word_id(lt_):
	
	word_pos = [-1]
	ln = [len(itm) for itm in lt_]
	for i in range(len(ln)):
		if i == 0:
			word_pos.append(ln[i]-1)
			continue
		else:
			word_pos.append(sum(ln[:i+1])-1)
	return word_pos


def get_tuple(lt_ori):

	tpl_ = []

	lt_w_id = get_word_id(lt_ori)

	for i in range(len(lt_w_id)):
		if i < len(lt_w_id) - 1:
			tpl_.append((lt_w_id[i],lt_w_id[i+1]))
	
	return tpl_


if __name__ == '__main__':
	
	t=Timer()
	t.start()

	standard = sys.argv[1]
	test_ = sys.argv[2]

	f_std = open(standard, "r", encoding='utf-8')
	f_tst = open(test_, "r", encoding='utf-8')

	senline, gold_seg, rslt_seg, rslt_seg_cor = 0, 0, 0, 0
	total_gold_seg, total_rslt_seg, total_cor_seg, total_incor_seg = 0, 0, 0, 0


	for eachl_std, eachl_tst in zip(f_std, f_tst):

		senline += 1

		#current line
		gold_seg = eachl_std.strip().split() #count words instead of position
		rslt_seg = eachl_tst.strip().split() # same for result
		
		# debugging
		#print(gold_seg)
		#print(rslt_seg)

		rslt_seg_cor = len(set(get_tuple(gold_seg)) & set(get_tuple(rslt_seg)))

		#debugging
		#print(rslt_seg_cor)
		
		total_gold_seg += len(gold_seg)
		total_rslt_seg += len(rslt_seg)
		total_cor_seg += rslt_seg_cor
	
	#incorrect segged words
	total_incor_seg = total_rslt_seg - total_cor_seg

	recall = float(total_cor_seg) / float(total_gold_seg) * 100
	precision = float(total_cor_seg) / float(total_rslt_seg) * 100

	up = 2 * precision * recall
	down = precision + recall
	f_value = float(up) / float(down)

	error_rate = float(total_incor_seg) / float(total_gold_seg) * 100

	print("The total line is:",senline)
	print("The total segged word in gold reference file is:",total_gold_seg)
	print("The total segged word in result file is:",total_rslt_seg)
	print("The total correct segged word in result file is:",total_cor_seg)
	print("The total incorrect segged word in result file is:",total_incor_seg)
	print("==============================================================")
	print("The evaluation result is as follow: \t")
	print("The Precision is:",round(precision,2))
	print("The Recall is:",format(recall,'.2f'))
	print("The F-measure is:",'%.2f'%(f_value))
	print("The Error rate is:",'%0.2f'%(error_rate))

	f_std.close()
	f_tst.close()

	print("The process was done in: {} s".format('%.2f'%(t.finish())))
