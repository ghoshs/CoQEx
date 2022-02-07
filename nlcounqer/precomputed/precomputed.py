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


def get_contexts(qtype, qid, source='bing'):
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


def get_entity_results(qtype, qid, contexts, **kwargs):
	entity_df = pd.read_csv(file_maps['enums_inverse'][qtype])
	entity_tuples = []  ## list(tuple(id, entity, score))
	entity_tuples_boost = []
	for item in contexts:
		entities = []
		cid = item['rank']
		enums = entity_df.loc[((entity_df['qid'] == qid) & (entity_df['cid'] == cid)), cols].to_dict('records')
		for enum in enums:
			if enum['answer'] not in item['context']:
				continue
			entity = {'answer': enum['answer'], 'entity': enum['entity'], 'score': enum['score']}
			entity['start'] = item['context'].index(enum['answer']) + enum['answer'].index(enum['entity'])
			entities.append(entity)
			entity_tuples.append((cid, enum['entity'], enum['score']))
			# entity_tuples_boost.append((cid, enum['entity'], enum['score_boost']))
		item['entities'] = entities
	return entity_tuples, contexts


def get_count_results(qtype, qid, contexts, **kwargs):
	count_df = pd.read_csv(file_maps['counts_inverse'][qtype])
	count_tuples = [] ## list(tuple(cardinal, score, id, text))
	count_tuples_boost = []
	count_data = []
	count_data_boost = []
	cardinals = [] ## list[tuple(cardinal, score)]
	cardinals_boost = []
	counts = count_df.loc[count_df['qid']==qid, cols].to_dict('records')
	try:
		count_tuples = ast.literal_eval(counts[0]['tf_answer'])
		cardinals = ast.literal_eval(counts[0]['cardinals'])
	except:
		pass
	finally:		
		count_tuples = [(item[3], round(item[1],2), int(item[0]), item[2]) for item in count_tuples] 
		cardinals = [(item[0], round(item[1],2)) for item in cardinals]
	# try: 
	# 	count_tuples_boost = ast.literal_eval(counts[0]['new_tf_answer'])
	# 	cardinals_boost = ast.literal_eval(counts[0]['new_cardinals'])
	# except:
	# 	pass
	# finally:
	# 	count_tuples_boost = [(item[3], round(item[1],2), int(item[0]), item[2]) for item in count_tuples_boost] 
	# 	cardinals_boost = [(item[0], round(item[1],2)) for item in cardinals_boost]

	print('count_tuples: ', count_tuples)
	prediction = None if len(count_tuples) == 0 else counts[0]['prediction']
	# prediction_boost = None if len(count_tuples_boost) == 0 else counts[0]['new_prediction']
	for item in contexts:
		cid = item['rank']
		count_span = {'text': '', 'start': 0, 'score': 0.0}
		cardinal = None
		for count in count_tuples:
			if count[2] == cid:
				count_span['score'] = count[1]
				count_span['text'] = count[3]
				count_span['start'] = item['context'].index(count[3])
				cardinal = count[0]
				if (count[0], count[1]) in cardinals:
					count_data.append(count)
					count_span['selected'] = True
		item['count_span'] = count_span
		item['cardinal'] = cardinal
		# for count in count_tuples_boost:
		# 	if count[2] == cid and (count[0], count[1]) in cardinals_boost:
		# 		count_data_boost.append(count)
	return prediction, count_data, contexts


def prepare_count_json(count_prediction, count_data, **kwargs):
	print('Count result to json')
	count_data_fdict = defaultdict(int)
	for num, score, id, text in count_data:
		count_data_fdict[num] += 1
	count_data_fdict = sorted(count_data_fdict.items(), key=lambda x:x[1], reverse=True)
	count_data_fdict_all = sorted(count_data, key=lambda x: x[1], reverse=True)	
	result = {
		'prediction': count_prediction,
		'all_data': [[item[3], item[1], item[2]+1] for item in count_data_fdict_all],
		'dataitems_freq': ', '.join([str(int(k)) + ' (' + str(v) + ')' for k, v in count_data_fdict]),
		'dataitems_sorted': ', '.join([str(int(item[0])) for item in sorted(count_data)]),
	}
	if 'old_data' in kwargs:
		# keep track of boosted data points.
		result['all_data'] = [[t, s, i+1, 0] if (c,s,i,t) in kwargs['old_data'] else [t, s, i+1, 1] for c,s,i,t in count_data_fdict_all]
	else:
		result['all_data'] = [[t, s, i+1] for c,s,i,t in count_data_fdict_all]
	return result


def prepare_enum_json(entity_data, **kwargs):
	print('Enumeration result to json')
	entity_fdict = defaultdict(int)
	entity_conf = defaultdict(list)
	for item in entity_data:
		entity_fdict[item[1]] += 1
		entity_conf[item[1]].append((item[2], item[0]))
	entity_fdict = sorted(entity_fdict.items(), key=lambda x: x[1], reverse=True)
	entity_conf = {k: sorted(v, key=lambda x: x[0], reverse=True)[0] for k, v in entity_conf.items()}
	result = {
		'entity_conf': [[k, v[0], v[1]+1] for k, v in sorted(entity_conf.items(), key=lambda x:x[1][0], reverse=True)],
		'entity_freq': ', '.join([k + ' (' + str(v) + ')' for k, v in entity_fdict]),
	}
	all_entity_conf = sorted(entity_data, key=lambda x:x[2], reverse=True)
	if 'old_data' in kwargs:
		result['all_entity_conf'] = [[e, s, i+1, 0] if (i,e,s) in kwargs['old_data'] else [e, s, i+1, 1] for i,e,s in all_entity_conf]
	else:
		result['all_entity_conf'] = [[e, s, i+1] for i,e,s in all_entity_conf]
	return result


def precomputed_queries(query, tfmodel, thresholds, aggregator):
	result = {}
	ticq = time.perf_counter()
	qtype = precomputed_query_type(query)
	print('Query type: ', qtype)
	qid = precomputed_query_id(query)
	
	print('Query ID: qid = ', qid)
	result['qtuples'] = get_qtuples(qtype, qid)
	
	print('Getting contexts .. ')
	contexts = get_contexts(qtype, qid)
	
	print('Getting enumerations .. ')
	# enum_cols = ['answer', 'entity', 'score', 'score_boost']
	# entity_data, entity_data_boost, contexts = get_entity_results(qtype, qid, contexts)
	enum_cols = {'answer': 'answer', 'entity': 'entity', 'score': 'score'}
	entity_data, contexts = get_entity_results(qtype, qid, contexts, cols=enum_cols)
	
	print('Getting counts .. ')
	# count_cols = ['tf_answer', 'cardinals', 'prediction', 'new_tf_answer', 'new_cardinals', 'new_prediction']
	# prediction, count_data, prediction_boost, count_data_boost, contexts = get_count_results(qtype, qid, contexts)
	count_cols = {'tf_answer': 'tf_answer', 'cardinals': 'cardinals', 'prediction': 'prediction'}
	prediction, count_data, contexts = get_count_results(qtype, qid, contexts, cols=count_cols)
	
	result['count'] = prepare_count_json(prediction, count_data) 
	# result['count_boost'] = prepare_count_json(prediction_boost, count_data_boost, old_data=count_data)
	result['entities'] = prepare_enum_json(entity_data) 
	# result['entities_boost'] = prepare_enum_json(entity_data_boost, old_data=entity_data)
	result['annotations'] = contexts
	toc = time.perf_counter()
	return result, toc - ticq

def prefetched_contexts(query):
	ticq = time.perf_counter()
	qtype = precomputed_query_type(query)
	print('Query type: ', qtype)
	qid = precomputed_query_id(query)

	print('Query ID: qid = ', qid)
	qtuples = get_qtuples(qtype, qid)
	
	print('Getting contexts .. ')
	contexts = get_contexts(qtype, qid)
	return contexts, qtuples