import sys
import pprint
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
	qtuples = get_qtuples(query, 'en_core_web_sm')
	
	### 2. Document retrieval (Bing/Wikipedia): JSON 
	results = call_bing_api(query, max_results)
		
	### 3. Count predictions
	count_prediction, count_data, count_annotations = predict_count(query, results, model, aggregator)

	### 4. Enumeration predction
	# predict_enumerations(query, qtuples, results)


	result['qtuples'] = {
		'type': qtuples.type, 
		'entity': ';'.join(qtuples.entity), 
		'relation': qtuples.relation, 
		'context': ';'.join(qtuples.context)
	}
	result['count_prediction'] = count_prediction
	result['count_data'] = [{'cardinal': item[0], 'score': item[1], 'id': item[2]} for item in count_data]
	result['count_annotations'] = count_annotations
	# for doc in results_tags:
	# 	ent_match = []
	# 	integers = []
	# 	text_cardinals = [{'text': ent.text, 'id': idx} for idx, ent in enumerate(doc.ents) if ent.label_ == 'CARDINAL']
	# 	doc_json = doc.to_json()
	# 	result['results_tags'].append({'text': doc_json['text'], 'ents': doc_json['ents']}) 
	return result