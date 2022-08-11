import sys
# sys.path.append('/')

import configparser
import random
from count_prediction.run_subprocess import run_subprocess
import re
import os
from tqdm import tqdm

# read api keys from config file
config = configparser.ConfigParser()
# config.read('//nlcounqer/count_prediction/count_config.ini')
### server edit
config.read('/nlcounqer/count_prediction/count_config_server.ini')

CogCompPath = config['cogcomp']['CogCompPath']
CogCompQuantifier = config['cogcomp']['Quantifier']
NormalizeQuant = config['cogcomp']['Normalize']
TmpDir = config['paths']['TmpDir']

quant_pattern = re.compile(r"(\[[^\]]*\])\s*(\[[^\]]*\]):(\(\d+,\s\d+\))")
ntuple_pattern = re.compile(r"\[(\S*)\s+(\d+(?:\.\d+)?)([Ee][+-]?\d+)?\s+(.*[^\s]?)\s*\]")


def structured_count(text, cogcomp_result, quant_pattern, ntuple_pattern):
	empty_extraction = []

	if len(cogcomp_result) == 0:
		# print("No extraction from: "+text)
		return empty_extraction

	all_patterns = re.findall(quant_pattern, cogcomp_result) ### [1] [2]:(3)

	if len(all_patterns) == 0:		
		# print("No patterns fromm: "+text)
		return empty_extraction
	
	# for pattern in all_patterns:
	all_ntuples = [re.findall(ntuple_pattern, pattern[1]) for pattern in all_patterns] ### [1 2 3 4] 
	all_ntuples = [ntuple[0] if len(ntuple)>0 else () for idx, ntuple in enumerate(all_ntuples)]
	spans = [pattern[2] for pattern in all_patterns]
	if len(all_ntuples) == 0:
		# print("No triples from: "+text+' : '+cogcomp_result)
		return empty_extraction
	else:
		extraction = list(zip(all_ntuples, spans))
		# print("Extracted: "+ str(extraction))
		return extraction

# return list of tuples
def get_cogcomp_ntuples(snippet):
	empty_extraction = []
	tempfile = TmpDir + 'tmp' + '_' + str(random.random())[2:] + '.txt'
	with open(tempfile, 'w', encoding='utf-8') as fp:
		fp.write(snippet)
	process_args = ['java', '-cp', CogCompPath, CogCompQuantifier, tempfile, NormalizeQuant]
	cogcomp_result = run_subprocess(process_args)
	os.remove(tempfile)
	structured_result = structured_count(snippet, cogcomp_result, quant_pattern, ntuple_pattern)
	if len(structured_result) == 0:
		return empty_extraction
	else:
		return structured_result

def get_count_spans(counts_in_snippet):
	count_spans = []
	for ntuple, span in counts_in_snippet:
		if len(ntuple) == 0:
			continue
		relation, quantity, exponent, metric = ntuple
		quantity += exponent
		# Equality
		# if query_answer_dict['answer.relation'] == '=' and len(query_answer_dict['answer.metric']) == 0 and relation in ['=', '~']:
			# gold_answer = float(query_answer_dict['answer.norm'])
			# snippet_count = float(quantity) 
			# ratio = gold_answer/snippet_count if snippet_count >= gold_answer else snippet_count/gold_answer
			# if ratio >= acceptance_threshold:
		indices = [index.strip() for index in span[1:-1].split(',')]
		count_spans.append({'relation': relation, 'quantity': str(float(quantity)), 'start': indices[0], 'end': indices[1]})

	return count_spans
