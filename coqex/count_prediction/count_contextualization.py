from sentence_transformers import util

def get_cnp_groups(prediction, sorted_data):
	"""
		return the representative cnp, 
		group1: cnps with count == prediction
		group2: other cnps
	"""
	cnp_rep, group1, group2 = None, [], []
	for cardinal, score, _id, text in sorted_data:
		if int(cardinal) == int(prediction):
			group1.append((cardinal, score, _id, text))
		else:
			group2.append((cardinal, score, _id, text))
	if len(group1)>0:
		group1 = sorted(group1, key=lambda x: float(x[1]), reverse=True)
		cnp_rep = group1[0]
		group1 = group1[1:] if len(group1)>1 else []
	return cnp_rep, group1, group2 


def get_cnp_classes(sbert, cnp_rep, cnps_group1, cnps_group2, prediction, cutoff=0.3):
	equivalent = []
	unrelated = []
	subgroup = []
	sorted_data = []
	if cnp_rep is not None:
		if len(cnps_group1) > 0:
			embedding_equivalent = sbert.encode([cnp_rep[3].lower()],convert_to_tensor=True)
			embeddings_group1 = sbert.encode([text.lower() for cardinal, score, _id, text in cnps_group1],convert_to_tensor=True)
			cosine_scores = util.pytorch_cos_sim(embedding_equivalent, embeddings_group1)
			for i in range(len(cnps_group1)):
				if cosine_scores[0][i] > 0:
					equivalent.append(cnps_group1[i])
				else:
					unrelated.append(cnps_group1[i])
		if len(cnps_group2) > 0:
			embedding_equivalent = sbert.encode([cnp_rep[3].lower()], convert_to_tensor=True)
			embeddings_group2 = sbert.encode([text.lower() for cardinal, score, _id, text in cnps_group2],convert_to_tensor=True)
			cosine_scores = util.pytorch_cos_sim(embedding_equivalent, embeddings_group2)
			for i in range(len(cnps_group2)):
				if cosine_scores[0][i] > 0:
					range_low = int((1-cutoff)*float(prediction))
					range_high = int((1+cutoff)*float(prediction))
					cardinal = int(cnps_group2[i][0])
					if cardinal >= range_low and cardinal <= range_high:
						equivalent.append(cnps_group2[i])
					elif cardinal < prediction:
						subgroup.append(cnps_group2[i])
					else:
						# if high cosime similarity keep as equivalent else incomparables
						if cosine_scores[0][i] > 0.5:
							equivalent.append(cnps_group2[i])
						else:
							unrelated.append(cnps_group2[i])
				else:
					unrelated.append(cnps_group2[i])
		sorted_data = [(cnp_rep[0], cnp_rep[1], cnp_rep[2], cnp_rep[3], 'cnprep')]
		for cardinal, score, _id, text in equivalent:
			sorted_data.append((cardinal, score, _id, text, 'synonym'))
		for cardinal, score, _id, text in subgroup:
			sorted_data.append((cardinal, score, _id, text, 'subgroup'))
		for cardinal, score, _id, text in unrelated:
			sorted_data.append((cardinal, score, _id, text, 'incomparable'))
	elif len(cnps_group2) > 0:
		"""
			when no good match for cnp rep all are incomparables
		"""
		for cardinal, score, _id, text in cnps_group2:
			sorted_data.append((cardinal, score, _id, text, 'incomparable'))
	return sorted_data


def count_contextualization(sbert, prediction, sorted_data, annotated_contexts):
	"""
		return count context tags -> representative, synonyms, subgroups, incomparables in the sorted data and annotated contexts
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
	# sorted_data, annotated_contexts = get_categories(sbert, prediction, sorted_data, annotated_contexts)
	cnp_rep, cnps_group1, cnps_group2 = get_cnp_groups(prediction, sorted_data)
	sorted_data = get_cnp_classes(sbert, cnp_rep, cnps_group1, cnps_group2, prediction)
	id_cnp_class = {_id: cnp_class for _, _, _id, _, cnp_class in sorted_data}
	for item in annotated_contexts:
		if 'count_span' in item and 'selected' in item['count_span'] and item['count_span']['selected']:
			if item['rank'] in id_cnp_class:
				item['count_span']['cnp'] = id_cnp_class[item['rank']]
			else:
				item['count_span']['cnp'] = None
	return sorted_data, annotated_contexts
