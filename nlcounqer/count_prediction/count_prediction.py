import json
from count_prediction.myw2n import word_to_num
from count_prediction.count_extraction import get_cogcomp_ntuples, get_count_spans
from count_prediction.apply_aggregator import apply_aggregator
import os
from os import path
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_noun_phrase_w_count(nlp, context, answer, start):
	ann = nlp(context)
	for chunk in ann.noun_chunks:
		if chunk.start_char <= start and chunk.end_char >= start+len(answer):
			return chunk.start_char, chunk.end_char
	return None, None


def predict_count(query, contexts, tfmodel, thresholds, aggregator, nlp):
	time_elapsed_prediction = 0
	time_elapsed_extraction = 0
	time_elapsed_aggregation = 0
	for item in contexts:
		## 1. span prediction 
		try:
			tic = time.perf_counter()
			answer = tfmodel(question=query, context=item['context'])
			time_elapsed_prediction += time.perf_counter() - tic
		except:
			answer = {'answer':'', 'score': 0, 'start': -1}
		finally:
			##2. Count extraction
			tic = time.perf_counter()
			item['count_span'] = {'text': answer['answer'], 'score': answer['score'], 'start': answer['start']}
			np_start, np_end = answer['start'], answer['start']+len(answer['answer'])
			# np_start, np_end = get_noun_phrase_w_count(nlp, item['context'], answer['answer'], answer['start'])
			if np_start is not None:
				item['count_span']['np_start'] = np_start
				item['count_span']['np_end'] = np_end
			cardinal = None
			try:
				cardinal = word_to_num(item['count_span']['text'])
			except ValueError:
				# print('%.3f\t%s' % (item['count_span']['score'], item['count_span']['text']))
				ntuple = get_cogcomp_ntuples(item['count_span']['text'])	
				count_span = get_count_spans(ntuple)
				cardinal = float(count_span[0]['quantity']) if len(count_span)>0 else None
			finally:
				item['cardinal'] = int(cardinal) if cardinal is not None and int(cardinal) > 0 else None 
			time_elapsed_extraction += time.perf_counter() - tic

	##3. Evaluate
	tic = time.perf_counter()
	prediction, sorted_data, annotated_contexts = apply_aggregator(contexts, aggregator, thresholds)
	time_elapsed_aggregation += time.perf_counter() - tic
	print('Prediction took %.4f secs\nExtraction took %.4f secs\nAggregation took %.4f secs'%(time_elapsed_prediction, time_elapsed_extraction, time_elapsed_aggregation))
	return prediction, sorted_data, annotated_contexts