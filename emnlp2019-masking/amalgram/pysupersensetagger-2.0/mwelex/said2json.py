#!/usr/bin/env python2.7
from __future__ import print_function
from nltk.tree import *
import json, itertools, sys, re

'''
Extracts MWE entries from the SAID idioms database.
JSON output will be written to stdout.

Arg: /path/to/said/data

@author: Emily Danchik
'''

path = sys.argv[1] + "/said4.txt"

with open(path) as f: # safer way to open files: if there is an exception it will still be properly closed
	lines = f.read().splitlines() # I tried to do f.readlines(), but apparently that's different! a unix vs. windows thing?
#f.close(): unnecessary with 'with'

def checkForMult(s):
	letters = list(s)
	rightCount = 0
	leftCount = 0
	
	for l in letters:
		if l == ')':
			rightCount = rightCount+1
		if l == '(':
			leftCount = leftCount+1
		if leftCount == rightCount and not leftCount == 1 and not l == len(letters):
			return "true"
	return "false" # Python literals are True and False (capitalized, no quotes)


def splitThese(s):
	letters = list(s)
	rightCount = 0
	leftCount = 0
	startHere = 0
	returnS = ""
	count = 0
	
	for l in letters:
		if l == ')':
			rightCount = rightCount+1
		if l == '(':
			leftCount = leftCount+1
		if leftCount == rightCount:
			if not letters.index(l) == len(letters):
				#returnS = returnS + s[startHere:letters.index(l)+1] + "***\n"
				returnS = returnS + s[startHere:count+1] + "***"
				#startHere = letters.index(l)+1	
				startHere = count+1
		count = count + 1
	return returnS
	
datasource = "SAID"
label = "MWE" # changed

tagset = set() # will contain all POS tags seen
ENTRIES = set()

for line in lines:
	line = line.replace("'", "$")
	if line[0] == '*':
		line = line[1:]
	#if line.find("e.g.") > -1:
	#	location = line.find("e.g.")
	#	line = line[:location]
	# deal with comments at the end of the line--including e.g./eg. entries, which illustrate possible lexical fillers
	line = re.sub(r'[^\)]+$', '', line)

	# (AUX(to)), (AUX(MOD ought)(to)), etc.
	line = line.replace('(to)','(TO to)')
	# missing space
	line = line.replace('PRONI', 'PRON I')
	# missing POSS
	line = line.replace("(NP $s)", "(POSS(NP) $s)")

	# lexically empty constituents get converted to slots
	line = re.sub(r'\(([A-Z]P?)\)',r'(\1 _sth_)', line)
	# ignore big PRO and weird empty CONJ
	line = line.replace('(PRO)','').replace('(CONJ)','')

	# list weird cases with lexically empty constituents
       	#if re.search(r'\((\S+)\)', line):
	#	print(line, file=sys.stderr)

	hold = splitThese(line)
	possibleMults = hold.split("***")
	for p0 in possibleMults:
		if p0.strip() in {'',','}: continue # only whitespace, or comma between analyses
		assert len(p0)>=2

		# commas don't have a POS tag
		p0 = p0.replace(',', '(, ,)')

		# input is buggy when a noun or adjective starts with a capital letter: space is off by one! NP is OK though.
		# problematic words: God, Jesus, Thomas, Harry, Daniel, Don Juan, Walter Mitty, Solomon, Big Top, Calcutta, Cain, East, West, Job, King Charles, Greek, Europe, King, Devil, Joneses, (blarney) Stone, Waterloo, Dutch, French
		p0 = re.sub(r'\([AN]([A-OQ-Z]) ', r'(N \1', p0)
		p0 = re.sub(r'( [a-zA-Z]+)([A-Z]) ', r'\1 \2', p0) # DonJ uan -> Don Juan, KingC harles -> King Charles

		# alternate word and tag, e.g. (DET a/POSS(NP(PRON one))'s)
		# add parentheses: ((DET a)/(POSS(NP(PRON one))'s))
		p0 = p0.replace('(DET the/POSS(NP(PRON our)))','((DET the)/(POSS(NP(PRON our))))').replace("(DET a/POSS(NP(PRON one))$s)","((DET a)/(POSS(NP(PRON one))$s))").replace("(DET an/POSS(NP(PRON one))$s)","((DET an)/(POSS(NP(PRON one))$s))").replace("(DET the/POSS(NP(PRON one))$s)","((DET the)/(POSS(NP(PRON one))$s))")

		# buggy instance
		p0 = p0.replace("(POSS(NP(N sb)$s))", "(POSS(NP(N sb))$s)")

		if ')/(' in p0: # slashes between phrases
			assert p0.count(')/(')==1
			islash = p0.index(')/(')+1

			# match the parenthesis before the slash
			nesting = 1
			for i in range(islash-2,-1,-1):
				if p0[i]==')':
					nesting += 1
				elif p0[i]=='(':
					nesting -= 1
				if nesting==0:
					break
			else: # never matched the parenthesis before the slash
				assert False,p0

			# match the parenthesis after the slash                                                      
                        nesting = 1
                        for j in range(islash+2,len(p0)):
                                if p0[j]=='(':
                                        nesting += 1
                                elif p0[j]==')':
			       	        nesting -= 1
				if nesting==0:
			                break
                        else: # never matched the parenthesis after the slash
				assert False,p0

			pvariants = [p0[:islash]+p0[j+1:], p0[:i]+p0[islash+1:]]
			
			#if 'head' in p0 and 'idea' in p0:
			#	print(p0,pvariants, file=sys.stderr)
		else:
			pvariants = [p0]

		if 'etc)' in p0 or 'etc.)' in p0: # used to indicate that a lexical item is just an example--synonyms are acceptable
			# two variants: one with the provided lexical item, one with a '_sth_' variable
			pvariants2 = []
			for pv in pvariants:
				pvariants2.extend([re.sub(r'\(([A-Z]+) ([^)]+) ?etc\.?\)', r'(\1 \2)', pv),
						   re.sub(r'\(([A-Z]+) ([^)]+) ?etc\.?\)', r'(\1 _sth_)', pv)])
			pvariants = pvariants2

		for p in pvariants:

			tree = Tree(p)

			tokenized = tree.pos()
			
			listOfWords = []
			for leaf in tokenized:
				startPoint = 0
				partOfList = []
				multiList = []
				if '/' in leaf[0]:
					partOfList = leaf[0].split('/')
					for eachWord in partOfList:
						multiList.append((eachWord, leaf[1]))
					listOfWords.append(multiList)
				else:
					partOfList = leaf[0]
					listOfWords.append((partOfList, leaf[1]))
			#print list(itertools.product(*listOfWords))
			#if 'head' in p0 and 'idea' in p0:
			#	print(p,tokenized,listOfWords, file=sys.stderr)

			y = [[]]

			for alternatives in listOfWords: # renamed word -> alternatives for clarity
				if isinstance(alternatives,tuple): # preferred way to check the type
					# only one alternative
					for sublist in y:
						sublist.append(alternatives) # no need to convert tuple to list
				else:
					#assert False,(alternatives,listOfWords) # a debugging trick
					y2 = []
					for tagged_word in alternatives:
						for sublist in y:
							addThis = sublist + [tagged_word] # list() converts the tuple to a list; instead we want to put the tuple in a list!
							y2.append(addThis)
					y = list(y2)
			
			for combo in y:
				# poses are the parts of speech, like this: ["V", "P"]
				
				#combo = str(combo) # converting a nice data structure to a string for processing invites bugs
				
				#importantBits = combo.split("\'")[1::2]
				#countIt = 0
				lemmas = []
				poses = []
				
				#for ib in importantBits:
				#print(combo)
				for j,(word,pos) in enumerate(combo): # enumerate() makes (offset,element) tuples. you can iterate over fixed-length tuples by specifying variables for their elements.
					if pos in {'PPRON','PRONN'}: # typos in data
						pos = 'PRON'
					elif pos=='POSS' and (lemmas[-1].endswith("'s_") or lemmas[-1]=='its'):
						continue # merged the possessive marker with previous token

					if word=='one' and pos=='PRON':
						word = '_sb_'
					elif word == "sb" or word == "sth":
						word = '_' + word + '_'
					elif word=='ing': # indicates any present participle verb
						word = '_VBG_'
						if pos=='VP':
							pos = 'V'
					elif word in {'cries','_sth_'} and pos=='VP': # buggy tag
						pos = 'V'
					elif word in {'oneself','self'}:
						word = '_oneself_'

					# merge in possessive marker
					if j+1<len(combo) and combo[j+1][1]=='POSS' and word[0]==word[-1]=='_':
						word = word[:-1]+"'s_"

					if word in {"$s","'s"}:
						assert pos in {'POSS','V','AUX','PRON'},combo

       					lemmas.append(word.replace("$", "'"))
					
					poses.append(pos)
					tagset.add(pos)

				# note that "poses" sometimes includes phrasal categories (NP, VP, AP)
				# where an entire phrase can be substituted
				entryJ = {"datasource": datasource, "label": label, "poses": tuple(poses), "words": tuple(lemmas)} # list-to-tuple conversion doesn't affect JSON, but makes it possible to cache the entry in a set
				if tuple(sorted(entryJ.items())) not in ENTRIES: # ensure uniqueness (some entries are redundant)
					print(json.dumps(entryJ))
					ENTRIES.add(tuple(sorted(entryJ.items())))
			
print(tagset, file=sys.stderr)
