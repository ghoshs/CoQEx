import sys
import pprint
from collections import defaultdict
#### server edit
# sys.path.append('/nlcounqer')
sys.path.append('//nlcounqer')

from retrieval.bing_search import call_bing_api
from query_model.query_model import get_qtuples
from count_prediction.count_prediction import predict_count
from enumeration_prediction.enumeration_prediction import predict_enumerations

def pipeline(query, max_results=10, model="default", aggregator="weighted"):
	result = {}
	
	### 1. Query modeling: namedtuple QTuples('type', 'entity', 'relation' 'context')
	print('Getting query tuples')
	qtuples = get_qtuples(query, 'en_core_web_sm')
	
	### 2. Document retrieval (Bing/Wikipedia): JSON 
	print('Retrieving relevant documents')
	results = call_bing_api(query, max_results)
		
	### 3. Count predictions
	print('Count prediction')
	count_prediction, count_data, results = predict_count(query, results, model, aggregator)
	count_data_fdict = defaultdict(int)
	for num, score, id, text in count_data:
		count_data_fdict[num] += 1
	count_data_fdict = sorted(count_data_fdict.items(), key=lambda x:x[1], reverse=True)
	count_data_fdict_all = sorted(count_data, key=lambda x: x[1], reverse=True)	

	### 4. Enumeration predction
	print('Enumeration prediction')
	entity_data, results = predict_enumerations(query, qtuples, results)
	entity_fdict = defaultdict(int)
	entity_conf = defaultdict(list)
	for item in entity_data:
		entity_fdict[item[1]] += 1
		entity_conf[item[1]].append((item[2], item[0]))
	entity_fdict = sorted(entity_fdict.items(), key=lambda x: x[1], reverse=True)
	entity_conf = {k: sorted(v, key=lambda x: x[0], reverse=True)[0] for k, v in entity_conf.items()}
	# entity_conf = [(k, v[0], []) for k, v in sorted(entity_conf.items(), key=lambda x: x[1][0])]
	result['qtuples'] = {
		'type': qtuples.type, 
		'entity': ';'.join(qtuples.entity), 
		'relation': qtuples.relation, 
		'context': ';'.join(qtuples.context)
	}
	result['count'] = {
		'prediction': count_prediction,
		'dataitems_freq': ', '.join([str(int(k)) + ' (' + str(v) + ')' for k, v in count_data_fdict]),
		'dataitems_sorted': ', '.join([str(int(item[0])) for item in count_data]),
		'all_data': ', '.join([item[3] + ' [' + str(item[1]) + ', ' + str(item[2]+1) + ']' for item in count_data_fdict_all])
	}
	# result['count_annotations'] = count_annotations
	result['entities'] = {
		'entity_freq': ', '.join([k + ' (' + str(v) + ')' for k, v in entity_fdict]),
		'entity_conf': ', '.join([k + ' [' + str(v[0]) + ', ' + str(v[1]+1) + ']' for k, v in sorted(entity_conf.items(), key=lambda x:x[1][0], reverse=True)]),
		'all_entity_conf': ', '.join([e + ' [' + str(s) + ', ' + str(i+1) + ']' for i, e, s in entity_data])
	}
	result['annotations'] = results
	return result