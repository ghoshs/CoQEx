from transformers import pipeline
import json
import configparser
from count_prediction.myw2n import word_to_num
from count_prediction.count_extraction import get_cogcomp_ntuples, get_count_spans
from count_prediction.apply_aggregator import apply_aggregator

config = configparser.ConfigParser()
config.read('//nlcounqer/count_prediction/count_config.ini')
### server edit
# config.read('/nlcounqer/count_prediction/count_config_server.ini')

model_path_dict = json.load(open(config['paths']['ModelPath'], 'r'))

def predict_count(query, contexts, model='default', aggregator='weighted'):
	model_path = model_path_dict[model]['model_path']
	thresholds = model_path_dict[model]['thresholds']
	tfmodel = pipeline("question-answering", model_path)
	for item in contexts:
		## 1. span prediction 
		try:
			answer = tfmodel(question=query, context=item['context'])
		except:
			answer = {'answer':'', 'score': 0, 'start': -1}
		finally:
			##2. Count extraction
			item['count_span'] = {'text': answer['answer'], 'score': answer['score'], 'start': answer['start']}
			cardinal = None
			try:
				cardinal = word_to_num(item['count_span']['text'])
			except ValueError:
				# print('%.3f\t%s' % (item['count_span']['score'], item['count_span']['text']))
				ntuple = get_cogcomp_ntuples(item['count_span']['text'])	
				count_span = get_count_spans(ntuple)
				cardinal = float(count_span[0]['quantity']) if len(count_span)>0 else None
			finally:
				item['cardinal'] = cardinal

	##3. Evaluate
	prediction, sorted_data, annotated_contexts = apply_aggregator(contexts, aggregator, thresholds)
	return prediction, sorted_data, annotated_contexts