import numpy as np
from collections import Counter
import random

PERCENTILE_LEVEL = 50

def prepare_data(contexts, threshold):
	cardinals, scores, ids, text = [], [], [], []
	reduced_threshold = threshold
	MINIMUM_CARDINALS = 5
	while len(cardinals) < MINIMUM_CARDINALS and reduced_threshold >= 0.0:
		for context in contexts:
			if 'selected' not in context['count_span'] or (not context['count_span']['selected']):
				if 'cardinal' in context and context['cardinal'] is not None and round(float(context['count_span']['score']),2) >= float(reduced_threshold) and int(context['cardinal'])>0:
					cardinals.append(int(context['cardinal']))
					scores.append(round(float(context['count_span']['score']), 2))
					ids.append(int(context['rank']))
					text.append(context['count_span']['text'])
					context['count_span']['selected'] = True
					context['count_span']['score'] = round(float(context['count_span']['score']),2)
				elif 'cardinal' in context and context['cardinal'] is not None:
					context['count_span']['selected'] = False
					context['count_span']['score'] = round(float(context['count_span']['score']),2)
		if len(cardinals) >= MINIMUM_CARDINALS:
			break
		reduced_threshold -= 0.1
	reduced_threshold = max(0, reduced_threshold)

	data = list(zip(np.array(cardinals), np.array(scores), np.array(ids), np.array(text, dtype=object)))
	return data, contexts, round(reduced_threshold, 2)


def get_weighted_prediction(data):
	if len(data) == 0:
		return 	None, data
	# cardinals, scores = np.array(cardinals), np.array(scores)
	sorted_cardinals, sorted_scores, sorted_ids, sorted_texts = map(np.array, zip(*sorted(data)))
	half_score = (PERCENTILE_LEVEL/100.0) * sum(sorted_scores)
	## in case of zero weights or single data
	if any(sorted_scores > half_score):
		median = (sorted_cardinals[sorted_scores == np.max(sorted_scores)])[0]
	else:
		cumsum_scores = np.cumsum(sorted_scores)
		mid_idx = np.where(cumsum_scores <= half_score)[0][-1]
		if cumsum_scores[mid_idx] == half_score:
			median = np.mean(sorted_cardinals[mid_idx:mid_idx+2])
		else:
			median = sorted_cardinals[mid_idx+1]
	return int(median), list(zip(sorted_cardinals.tolist(), 
		sorted_scores.tolist(), sorted_ids.tolist(), sorted_texts.tolist()))


def get_median_prediction(data):
	if len(data) == 0:
		return None, data
	else:
		sorted_cardinals, sorted_scores, sorted_ids, sorted_texts = map(np.array, zip(*sorted(data))) 		
		median = np.percentile(sorted_cardinals, PERCENTILE_LEVEL, interpolation='higher')
		return int(median), list(zip(sorted_cardinals.tolist(), 
				sorted_scores.tolist(), sorted_ids.tolist(), sorted_texts.tolist()))


def get_max_prediction(data):
	if len(data) == 0:
		return None, data
	else:
		sorted_cardinals, sorted_scores, sorted_ids, sorted_texts = map(np.array, 
			zip(*sorted(data, key=lambda x: x[1], reverse=True))) 
		return int(sorted_cardinals[0]), list(zip(sorted_cardinals.tolist(), 
			sorted_scores.tolist(), sorted_ids.tolist(),sorted_texts.tolist()))

def get_frequent_prediction(data):
	if len(data) == 0:
		return None, data
	else:
		sorted_cardinals, sorted_scores, sorted_ids, sorted_texts = map(np.array, zip(*sorted(data))) 
		cardinal_dict = dict(Counter(cardinal for cardinal in sorted_cardinals))
		highest_frequency = sorted(cardinal_dict.items(), reverse=True, key=lambda x:x[1])[0][1]
		random.seed(10)
		frequent = random.choice([cardinal for cardinal, freq in cardinal_dict.items() if freq == highest_frequency])
		return frequent, list(zip(sorted_cardinals.tolist(), 
			sorted_scores.tolist(), sorted_ids.tolist(),sorted_texts.tolist()))


def apply_aggregator(contexts, aggregator, threshold):
	data, annotated_contexts, reduced_threshold = prepare_data(contexts, threshold)
	if aggregator == 'weighted':
		prediction, sorted_data = get_weighted_prediction(data)
		print('Weighted Prediction: ', prediction)
		print('Sorted data: ', sorted_data)
	elif aggregator == 'median':
		prediction, sorted_data = get_median_prediction(data)
		print('Median Prediction: ', prediction)
		print('Sorted data: ', sorted_data)
	elif aggregator == 'max':
		prediction, sorted_data = get_max_prediction(data)
		print('Max Prediction: ', prediction)
		print('Sorted data: ', sorted_data)
	elif aggregator == 'frequent':
		prediction, sorted_data = get_frequent_prediction(data)
		print('Frequent Prediction: ', prediction)
		print('Sorted data: ', sorted_data)
	else:
		prediction, sorted_data = None, None
	return prediction, sorted_data, annotated_contexts, reduced_threshold