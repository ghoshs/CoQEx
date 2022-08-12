import pandas as pd
from precomputed.query import precomputed_query_type, precomputed_query_id, file_maps
from query_model.query_model import QTuples
import ast
import json
import time
from collections import defaultdict


# path = '//nlcounqer/'
## server edit ##
path = '/nlcounqer/'
PRECOMPDATA = '/nlcounqer/static/data/precomputed/precomputed.jsonl'

"""
return precomputed results on CoQuAD using CoQEx
"""

def get_qtuples(qtype, qid):
	query_df = pd.read_csv(file_maps['query'][qtype])
	tuples = query_df.loc[query_df['qid']==qid, ['answer_type','query_entity','query_relation','query_context']].to_dict('records')[0]
	if not pd.isnull(tuples['query_context']):
		context = ast.literal_eval(tuples['query_context'])
		context = tuple(word for word in context)
	else:
		context = ''
	if not pd.isnull(tuples['query_relation']):
		relation = tuples['query_relation']
	else:
		relation = ''
	if not pd.isnull(tuples['answer_type']):
		_type = tuples['answer_type']
	else:
		_type = ''
	if not pd.isnull(tuples['query_entity']):
		entity = ast.literal_eval(tuples['query_entity'])
		entity = tuple(ent for ent in entity)
	qtuples = QTuples(type=_type, entity=entity, relation=relation, context=context)
	return qtuples


def get_contexts(qtype, qid, query='', source='bing'):
	contexts = []
	with open(file_maps['snippets_'+source][qtype]) as fp:
		data = json.load(fp)
		for para in data['data'][0]['paragraphs']:
			_qid = int(para['qas'][0]['id'].split('_')[0])
			cid = int(para['qas'][0]['id'].split('_')[1])
			if qid != _qid:
				continue
			contexts.append({'context': para['context'], 'rank': cid, 'qid': para['qas'][0]['id']})
	return contexts


def prefetched_contexts(query):
	ticq = time.perf_counter()
	qtype = precomputed_query_type(query)
	print('Query type: ', qtype)
	qid = precomputed_query_id(query)

	print('Query ID: qid = ', qid)
	qtuples = get_qtuples(qtype, qid)
	
	print('Getting contexts .. ')
	contexts = get_contexts(qtype, qid, query)
	return contexts, qtuples


def get_precomputed_result(query):
	tic = time.perf_counter()
	response = None
	with open(PRECOMPDATA, 'r', encoding='utf-8') as fp:
		for line in fp.readlines():
			try:
				q, r = list(json.loads(line).items())[0]
			except:
				q, r = None, {}
			finally:
				if q == query:
					print("Found response for query; ", q, query)
					response = r
					break
	time_elapsed = time.perf_counter() - tic
	if response is not None:
		return response, time_elapsed
	else:
		return {}, time_elapsed
