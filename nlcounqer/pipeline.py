import sys
import pprint
from collections import defaultdict
import time
#### server edit
sys.path.append('/nlcounqer')
# sys.path.append('//nlcounqer')

from retrieval.bing_search import call_bing_api
from query_model.query_model import get_qtuples
from count_prediction.count_prediction import predict_count
from enumeration_prediction.enumeration_prediction import predict_enumerations
from interaction.interaction import boost_predictions

def prepare_count_json(count_prediction, count_data, **kwargs):
	print('Count result to json')
	# ticcj = time.perf_counter()
	count_data_fdict = defaultdict(int)
	for num, score, id, text in count_data:
		count_data_fdict[num] += 1
	count_data_fdict = sorted(count_data_fdict.items(), key=lambda x:x[1], reverse=True)
	count_data_fdict_all = sorted(count_data, key=lambda x: x[1], reverse=True)	
	# toc = time.perf_counter()
	# print("Completed in %.4f secs."%(toc - ticcj))
	result = {
		'prediction': count_prediction,
		# 'dataitems_freq': count_data_fdict,
		# 'dataitems_sorted': count_data,
		'all_data': [[item[3], item[1], item[2]+1] for item in count_data_fdict_all],
		'dataitems_freq': ', '.join([str(int(k)) + ' (' + str(v) + ')' for k, v in count_data_fdict]),
		'dataitems_sorted': ', '.join([str(int(item[0])) for item in count_data]),
		# 'all_data': ', '.join([item[3] + ' [' + str(item[1]) + ', ' + str(item[2]+1) + ']' for item in count_data_fdict_all])
	}
	if 'old_data' in kwargs:
		result['all_data'] = [[t, s, i+1, 0] if (c,s,i,t) in kwargs['old_data'] else [t, s, i+1, 1] for c,s,i,t in count_data_fdict_all]
	else:
		result['all_data'] = [[t, s, i+1] for c,s,i,t in count_data_fdict_all]
	return result


def prepare_enum_json(entity_data, **kwargs):
	print('Enumeration result to json')
	# ticej = time.perf_counter()
	entity_fdict = defaultdict(int)
	entity_conf = defaultdict(list)
	for item in entity_data:
		entity_fdict[item[1]] += 1
		entity_conf[item[1]].append((item[2], item[0]))
	entity_fdict = sorted(entity_fdict.items(), key=lambda x: x[1], reverse=True)
	# entity_conf = {'entity_label': (score, id)}
	entity_conf = {k: sorted(v, key=lambda x: x[0], reverse=True)[0] for k, v in entity_conf.items()}
	# toc = time.perf_counter()
	# print("Completed in %.4f secs."%(toc - ticej))
	result = {
		# 'entity_freq': entity_fdict,
		'entity_conf': [[k, v[0], v[1]+1] for k, v in sorted(entity_conf.items(), key=lambda x:x[1][0], reverse=True)],
		# 'all_entity_conf': [[e, s, i+1] for i,e,s in entity_data],
		'entity_freq': ', '.join([k + ' (' + str(v) + ')' for k, v in entity_fdict]),
		# 'entity_conf': ', '.join([k + ' [' + str(v[0]) + ', ' + str(v[1]+1) + ']' for k, v in sorted(entity_conf.items(), key=lambda x:x[1][0], reverse=True)]),
		# 'all_entity_conf': ', '.join([e + ' [' + str(s) + ', ' + str(i+1) + ']' for i, e, s in entity_data])
	}
	if 'old_data' in kwargs:
		# result['entity_conf'] = [[k, v[0], v[1]+1, 0] for k, v in sorted(entity_conf.items(), key=lambda x:x[1][0], reverse=True)],
		result['all_entity_conf'] = [[e, s, i+1, 0] if (i,e,s) in kwargs['old_data'] else [e, s, i+1, 1] for i,e,s in entity_data]
	else:
		result['all_entity_conf'] = [[e, s, i+1] for i,e,s in entity_data]
	return result


def pipeline(query, tfmodel, thresholds, qa_enum, nlp, aggregator, max_results):
	result = {}
	
	### 1. Query modeling: namedtuple QTuples('type', 'entity', 'relation' 'context')
	print('Getting query tuples')
	ticq = time.perf_counter()
	qtuples = get_qtuples(query, nlp)
	toc = time.perf_counter()
	print("Completed in %.4f secs."%(toc - ticq))
	
	### 2. Document retrieval (Bing/Wikipedia): JSON 
	print('Retrieving relevant documents')
	ticr = time.perf_counter()
	## results -> list(dict(
	# 						rank,
	# 						url,
	# 						about,
	# 						context,
	# 						dateLastCrawled))
	results = call_bing_api(query, max_results)
	toc = time.perf_counter()
	print("Completed in %.4f secs."%(toc - ticr))
	
	### 3. Count predictions
	print('Count prediction')
	ticc = time.perf_counter()

	## count_prediction -> int
	## count_data -> list(tuple(cardinal, score, id, text))
	## results -> list(dict(
	# 						rank,
	# 						url,
	# 						about,
	# 						context,
	# 						dateLastCrawled,
	# 						cardinal,
	# 						count_span: dict(selected, text, score)))
	count_prediction, count_data, results = predict_count(query, results, tfmodel, thresholds, aggregator, nlp)
	toc = time.perf_counter()
	print("Completed in %.4f secs."%(toc - ticc))
	
	### 4. Enumeration predction
	print('Enumeration prediction')
	tice = time.perf_counter()
	## entity_data -> list(tuple(id, entity, score))
	## results -> list(dict(
	# 						rank,
	# 						url,
	# 						about,
	# 						context,
	# 						dateLastCrawled,
	# 						cardinal,
	# 						count_span: dict(selected, text, score)
	# 						entities: dict(entity_text: 
	# 										dict(score, start, answer))))
	entity_data, results = predict_enumerations(query, qtuples, results, qa_enum, nlp)
	toc = time.perf_counter()
	print("Completed in %.4f secs."%(toc - tice))
	
	### 5. Interaction
	# print('Interaction ')
	# tici = time.perf_counter()
	# count_prediction_boost, count_data_boost, entity_data_boost = boost_predictions(count_prediction, count_data, aggregator, entity_data)
	# toc = time.perf_counter()
	# print("Completed in %.4f secs."%(toc - tice))

	result['qtuples'] = {
		'type': qtuples.type, 
		'entity': ';'.join(qtuples.entity), 
		'relation': qtuples.relation, 
		'context': ';'.join(qtuples.context)
	}
	result['count'] = prepare_count_json(count_prediction, count_data) 
	# result['count_boost'] = prepare_count_json(count_prediction_boost, count_data_boost, old_data=count_data)
	result['entities'] = prepare_enum_json(entity_data) 
	# result['entities_boost'] = prepare_enum_json(entity_data_boost, old_data=entity_data)
	result['annotations'] = results

	toc = time.perf_counter()
	print('Total time lapsed %.4f secs'%(toc-ticq))
	return result, toc - ticq