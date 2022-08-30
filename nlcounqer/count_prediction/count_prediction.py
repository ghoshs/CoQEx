import json
from count_prediction.myw2n import word_to_num
from count_prediction.count_extraction import get_cogcomp_ntuples, get_count_spans
from count_prediction.apply_aggregator import apply_aggregator
from count_prediction.count_contextualization import count_contextualization
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


def predict_count(query, contexts, tfmodel, threshold, aggregator, nlp, sbert):
	"""
		Returns the following variables
		prediction -> int
		sorted_data -> list(tuple(cardinal, score, id, text, context_class))
		annotated_contexts -> list(dict(
							rank,
							url,
							about,
							context,
							dateLastCrawled,
							cardinal,
							count_span: dict(selected, text, score, context_class)))
	"""
	time_elapsed_prediction = 0
	time_elapsed_extraction = 0
	time_elapsed_aggregation = 0
	time_elapsed_contextualization = 0
	countqa_contexts = [item['context'] for item in contexts if len(item['context'])>0]
	
	# with open('/nlcounqer/debug/contexts.json', 'w', encoding='utf-8') as fp:
	# 	json.dump(countqa_contexts, fp)

	## 1. span prediction 
	try:
		tic = time.perf_counter()
		countqa_pred = tfmodel(question=[query]*len(countqa_contexts), context=countqa_contexts, handle_impossible_answer=True)
	except:
		countqa_pred = []
	finally:
		time_elapsed_prediction = time.perf_counter() - tic
	
	pred_idx=0
	for item in contexts:
			if len(countqa_pred) == 0 or len(item['context']) == 0:
				answer = {'answer':'', 'score': 0, 'start': -1}
			else:
				answer = countqa_pred[pred_idx]
				pred_idx += 1

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
	prediction, sorted_data, annotated_contexts, reduced_threshold = apply_aggregator(contexts, aggregator, threshold)
	time_elapsed_aggregation += time.perf_counter() - tic

	##4. Classify Count Contexts
	tic = time.perf_counter()
	sorted_data, annotated_contexts = count_contextualization(sbert, prediction, sorted_data, annotated_contexts)
	time_elapsed_contextualization = time.perf_counter() - tic

	print('Prediction took %.4f secs\nExtraction took %.4f secs\nAggregation took %.4f secs\nContextualization took %.4f secs'%\
		(time_elapsed_prediction, time_elapsed_extraction, time_elapsed_aggregation, time_elapsed_contextualization))
	return prediction, sorted_data, annotated_contexts, reduced_threshold