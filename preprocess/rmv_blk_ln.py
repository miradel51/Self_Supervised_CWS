import sys
import pdb

origin_src = sys.argv[1]
removed_src = sys.argv[2]

origin_src_f = open(origin_src,"r")
removed_src_f = open(removed_src,"w")


cont_src = ""
cont_trg =""
senln = 0
blankln = []

#def rm_duplicate(origin_src,origin_trg):

for eachl_src in origin_src_f:
	senln += 1
	cont_src = eachl_src.strip()

	#check out whether were there still exist any blank lines
	if len(cont_src) < 1 :
		blankln.append(senln)
	else:
		removed_src_f.write(cont_src)
		removed_src_f.write("\n")

print("There were {0} blank line(s), they are {1}".format(len(blankln), blankln))
origin_src_f.close()
removed_src_f.close()
