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
	for num, score, id, text, cnp_class in count_data:
		count_data_fdict[num] += 1
	# count_data_fdict = sorted(count_data_fdict.items(), key=lambda x:x[1], reverse=True)
	# count_data_fdict_all = sorted(count_data, key=lambda x: x[1], reverse=True)	
	# toc = time.perf_counter()
	# print("Completed in %.4f secs."%(toc - ticcj))
	result = {
		'prediction': count_prediction,
		'all_count': [[item[3], round(item[1],2), item[2]+1, item[0], count_data_fdict[item[0]], item[4]] for item in count_data],
	}
	return result


def prepare_enum_json(entity_data, **kwargs):
	print('Enumeration result to json')
	# ticej = time.perf_counter()
	entity_fdict = defaultdict(int)
	entity_conf = defaultdict(list)
	for item in entity_data:
		entity_fdict[item[1]] += 1
		# entity_conf[item[1]].append((item[4], item[0]))
	# entity_fdict = sorted(entity_fdict.items(), key=lambda x: x[1], reverse=True)
	# entity_conf = {k: sorted(v, key=lambda x: x[0], reverse=True)[0] for k, v in entity_conf.items()}
	# toc = time.perf_counter()
	# print("Completed in %.4f secs."%(toc - ticej))
	result = {
		# 'all_entity_conf': [[e, s, i+1] for i,e,s in entity_data],
		# 'entity_freq': ', '.join([k + ' (' + str(v) + ')' for k, v in entity_fdict]),
		'all_entity': [[entity, round(score,2), _id+1, start, round(conf,2), entity_fdict[entity]] for _id,entity,conf,start,score in entity_data]
	}
	return result


def pipeline(query, tfmodel, thresholds, qa_enum, nlp, sbert, aggregator, max_results):
	result = {}
	
	### 1. Query modeling: namedtuple QTuples('type', 'entity', 'relation' 'context')
	print('Getting query tuples')
	ticq = time.perf_counter()
	qtuples = get_qtuples(query, nlp)
	print("Completed in %.4f secs."%(time.perf_counter() - ticq))
	
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
	print("Completed in %.4f secs."%(time.perf_counter() - ticr))
	
	### 3. Count predictions
	print('Count prediction')
	ticc = time.perf_counter()
	## count_prediction -> int
	## count_data -> list(tuple(cardinal, score, id, text, context_class))
	## results -> list(dict(
	# 						rank,
	# 						url,
	# 						about,
	# 						context,
	# 						dateLastCrawled,
	# 						cardinal,
	# 						count_span: dict(selected, text, score, context_class)))
	count_prediction, count_data, results = predict_count(query, results, tfmodel, thresholds, aggregator, nlp, sbert)
	print("Completed in %.4f secs."%(time.perf_counter() - ticc))

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
	print("Completed in %.4f secs."%(time.perf_counter() - tice))
	
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