#!/usr/bin/python3  
#author: Fan Luo

import numpy as np
import string 


#r1_pmids, r2_pmids are list of pmids
r1_pmids= np.loadtxt(open("sorted_positive_path_move_r3.csv", "rb"),skiprows=1,delimiter='[',dtype=str,usecols=(1,))
r2_pmids= np.loadtxt(open("sorted_positive_path_move_r3.csv", "rb"),skiprows=1,delimiter='[',dtype=str,usecols=(2,))
citations= np.loadtxt(open("citations_uniq.csv", "rb"),dtype=str)

with open('citation_features_positive.csv', 'w') as output:

# r1_pmids= np.loadtxt(open("sorted_all_negative_path_move_c1c2c3.csv", "rb"),skiprows=1,delimiter='[',dtype=str,usecols=(1,))
# r2_pmids= np.loadtxt(open("sorted_all_negative_path_move_c1c2c3.csv", "rb"),skiprows=1,delimiter='[',dtype=str,usecols=(2,))
# citations= np.loadtxt(open("citations_uniq.csv", "rb"),dtype=str)

# with open('citation_features_negative.csv', 'w') as output:

	for l, r1_pmid in enumerate(r1_pmids):
		pmids1 = r1_pmid.split('"')[1::2]
		pmids2 = r2_pmids[l].split(']')[0].split('"')[1::2]
		citation_count = 0
		same_count = 0

		for pmid1 in pmids1: 
			for pmid2 in pmids2: 
				p1_p2 = pmid1+','+pmid2
				p2_p1 = pmid2+','+pmid1
				if(p1_p2 in citations):
					citation_count += 1
				if(p2_p1 in citations):
					citation_count += 1
				if((pmid1 == pmid2) and (pmid1 != 0)):
					same_count += 1
		citation_feature = citation_count / (len(pmids1) + len(pmids2))
		Jaccard = same_count / (len(pmids1) + len(pmids2) - same_count)
		output.write(str(citation_feature))
		output.write(',')
		output.write(str(Jaccard))
		output.write('\n')
