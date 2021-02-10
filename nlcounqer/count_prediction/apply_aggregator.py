import numpy as np

def prepare_data(contexts, threshold):
	cardinals = []
	scores = []
	ids = []
	for context in contexts:
		if 'cardinal' in context and context['cardinal'] is not None and context['answer']['score'] >= threshold:
			cardinals.append(float(context['cardinal']))
			scores.append(float(context['answer']['score']))
			ids.append(int(context['rank']))
	data = list(zip(np.array(cardinals), np.array(scores), np.array(ids)))
	return data


def get_weighted_prediction(data):
	ptile = 50
	if len(data) == 0:
		return 	None, data
	# cardinals, scores = np.array(cardinals), np.array(scores)
	sorted_cardinals, sorted_scores, sorted_ids = map(np.array, zip(*sorted(data)))
	half_score = (ptile_level/100.0) * sum(sorted_scores)
	if any(scores > half_score):
		median = (cardinals[scores == np.max(scores)])[0]
	else:
		cumsum_scores = np.cumsum(sorted_scores)
		mid_idx = np.where(cumsum_scores <= half_score)[0][-1]
		if cumsum_scores[mid_idx] == half_score:
			median = np.mean(sorted_cardinals[mid_idx:mid_idx+2])
		else:
			median = sorted_cardinals[mid_idx+1]
	return median, list(zip(sorted_cardinals, sorted_scores, sorted_ids))


def get_median_prediction(data):
	ptile = 50
	if len(data) == 0:
		return None, data
	else:
		sorted_cardinals, sorted_scores, sorted_ids = map(np.array, zip(*sorted(data))) 		
		median = np.percentile(sorted_cardinals, ptile_level, interpolation='higher')
		return median, list(zip(sorted_cardinals, sorted_scores, sorted_ids))
	

def apply_aggregator(contexts, aggregator, thresholds):
	if aggregator == 'weighted':
		prediction, data = get_weighted_prediction(prepare_data(contexts, thresholds[aggregator]))
	elif aggregator == 'median':
		prediction, data = get_median_prediction(prepare_data(contexts, thresholds[aggregator]))
	return prediction, data