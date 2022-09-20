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

def prepare_count_json(count_prediction, count_data, **kwargs):
	print('Count result to json')

	count_data_fdict = defaultdict(int)
	prediction = None
	for num, score, id, text, cnp_class in count_data:
		count_data_fdict[num] += 1
		if num == count_prediction and cnp_class == 'cnprep':
			prediction = [text, round(score, 2), id+1, num]
	result = {
		'prediction': prediction,
		'all_count': [[item[3], round(item[1],2), item[2]+1, item[0], count_data_fdict[item[0]], item[4]] for item in count_data]
	}
	for arg in kwargs:
		result[arg] = kwargs[arg]
	return result


def prepare_enum_json(entity_data, **kwargs):
	print('Enumeration result to json')
	
	result = {'all_entity': []}
	for canon_entity in entity_data:
		for cid, entity, start, answer, conf, entailment in entity_data[canon_entity]['ann']:
			result['all_entity'].append([entity, 
				canon_entity,
				cid+1,
				start,
				round(conf,2), round(entailment,2), 
				round(entity_data[canon_entity]['type_compatibility_score'],2),
				round(entity_data[canon_entity]['answer_confidence_score'],2),
				round(entity_data[canon_entity]['context_frequency_score'],2),
				round(entity_data[canon_entity]['winning_document_score'],2),
				])
	
	for arg in kwargs:
		result[arg] = kwargs[arg]

	return result


def pipeline(query, tfmodel, count_threshold, qa_enum, enum_threshold, typepredictor, nlp, sbert, aggregator, max_results, **kwargs):
	result = {}
	
	### 1. Query modeling: namedtuple QTuples('type', 'entity', 'relation' 'context')
	print('Getting query tuples')
	ticq = time.perf_counter()
	if 'qtuples' in kwargs:
		qtuples = kwargs['qtuples']
	else:
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
	if 'contexts' in kwargs:
		results = kwargs['contexts']
	else:
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
	count_prediction, count_data, results, reduced_threshold_count = predict_count(query, 
		results, 
		tfmodel, 
		count_threshold, 
		aggregator, 
		nlp, sbert
	)
	print("Completed in %.4f secs."%(time.perf_counter() - ticc))

	### 4. Enumeration predction
	print('Enumeration prediction')
	tice = time.perf_counter()
	## entity_data -> dict(
	# 						canon_entity: 
	# 						dict(
	# 							ann: tuple(cid, entity, start, answer, conf, entailment),
	# 							type_compatibility_score: float,
	# 							answer_confidence_score: float,
	# 							context_frequency_score: float,
	# 							winning_document_score: float
	# 					  	)
	# 				  )
	## results -> list(dict(
	# 						rank,
	# 						url,
	# 						about,
	# 						context,
	# 						dateLastCrawled,
	# 						cardinal,
	# 						count_span: dict(selected, text, score, context_class)
	# 						entities: dict(entity_text: 
	# 										dict(selected, score, start, answer, entity, canonical))))
	entity_data, results, reduced_threshold_enum = predict_enumerations(
		query, 
		qtuples, 
		results, 
		qa_enum, 
		nlp, 
		typepredictor, 
		span_threshold=enum_threshold
	)
	print("Completed in %.4f secs."%(time.perf_counter() - tice))
	result['qtuples'] = {
		'type': qtuples.type, 
		'entity': ';'.join(qtuples.entity), 
		'relation': qtuples.relation, 
		'context': ';'.join(qtuples.context)
	}
	## result['count'] -> dict(
	# 						'prediction': float,
	# 						'all_count': list(list(
	# 										[text, score, id+1, cardinal, cardinal_frequency, context_class]
	# 									 ))
	# 					  )
	## result['entities'] -> dict('all_entity': list(list(
	# 								[entity, canon_entity, 
	# 								id, start, 
	# 								confidence, entailment, 
	# 								typecom_score, ansconf_score, freq_score, windoc_score]
	# 							  )))
	result['count'] = prepare_count_json(count_prediction, count_data, reduced_threshold=reduced_threshold_count) 
	result['entities'] = prepare_enum_json(entity_data, reduced_threshold=reduced_threshold_enum) 
	result['annotations'] = results

	toc = time.perf_counter()
	print('Total time lapsed %.4f secs'%(toc-ticq))
	return result, toc - ticq