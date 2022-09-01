'''
called from nlcounqer/pipeline.py
return values:
## entity_data -> list(tuple(id, entity, conf, start, score_winning, score_frequent, score_conf, score_typecom))
## results -> list(dict(
# 						rank,
# 						url,
# 						about,
# 						context,
# 						dateLastCrawled,
# 						cardinal,
# 						count_span: dict(selected, text, score, context_class)
# 						entities: dict(entity_text: 
# 										dict(selected, score, start, answer))))
'''

import re
import os
import time
import nltk
from nltk.corpus import stopwords
from typing import NamedTuple

stopwords_en = stopwords.words('english')


class EntailmentLabelProbs(NamedTuple):
	entailment: float = 0
	contradiction: float = 0
	neutral: float = 0


def pseudo_canonicalization(entity, stopwords):
	entity = entity.lower()
	no_stopwords = [word for word in entity.split() if word not in stopwords]
	return ' '.join(no_stopwords)


def get_model_predictions(qa, query, contexts, topk):
	predictions = {}
	for item in contexts:
		if query.startswith('How many ') or query.startswith('how many '):
			query = 'Which ' + query[len('how many '):]
		if query.lower().startswith('number of '):
			query = 'Which ' + query[len('number of '):]
		try:
			answer = qa(question=query, context=item['context'], top_k=topk)
		except:
			answer = []
		finally:
			predictions[item['rank']] = answer
	return predictions


def get_entities_spacy(contexts, predictions, nlp, span_threshold):
	numeric_classes = ['DATE', 'PERCENT', 'TIME', 'MONEY', 'CARDINAL', 'QUANTITY', 'ORDINAL']
	canon_entities, entities_per_context, doc_scores = {}, {}, {}
	MINIMUM_CANON_ENTITIES = 5 # same as MINIMUM_CARDINALS in count_prediction.apply_aggregator.prepare_data
	num_canon_entities, reduced_threshold = None, None
	for context in contexts:
		_id = context['rank']
		entities = {}
		if type(predictions[_id]) == dict: #model gives only one answer
			predictions[_id] = [predictions[_id]]
		for answer in predictions[_id]:
			ann = nlp(answer['answer'])
			mentions = [mention for mention in ann.ents if mention.label_ not in numeric_classes]
			for mention in mentions:
				entity_startchar = mention.start_char
				canon_entity = pseudo_canonicalization(mention.text, stopwords_en)
				entity = { 
						  'score': round(float(answer['score']),2), 
						  'start': answer['start'] + entity_startchar,
						  'answer': answer['answer'],
						  'entity': mention.text,
						  'canonical': canon_entity
						 }
				entity['selected'] = entity['score']>=span_threshold
				# todo: check containment 
				if canon_entity in entities and entity['start'] == entities[canon_entity]['start']:
					if entity['score'] > entities[canon_entity]['score']:
						entities[canon_entity]['score'] = entity['score']
				else:
					entities[canon_entity] = entity
				if canon_entity not in canon_entities:
					canon_entities[canon_entity] = {'ann': []}
				if entity['score'] >= span_threshold:
					canon_entities[canon_entity]['ann'].append([_id, entity['entity'], entity['start'], entity['answer'], entity['score']])
				if _id not in doc_scores or (_id in doc_scores and doc_scores[_id] < entity['score']):
					doc_scores[_id] = entity['score']
		
		## highest score per context
		context['entities'] = sorted([v for k, v in entities.items()], key=lambda x:x['start'])
		
	# reduced threshold setting if num_entities less than minimum entities
	# num_entities = sum([entity['selected'] for context in contexts for entity in context['entities']])
	num_canon_entities = sum([int(len(canon_entities[ent]['ann'])>0) for ent in canon_entities])
	if num_canon_entities >= MINIMUM_CANON_ENTITIES:
		reduced_threshold = span_threshold
	else:
		reduced_threshold = span_threshold - 0.1

	while num_canon_entities < MINIMUM_CANON_ENTITIES and reduced_threshold >= 0:
		for context in contexts:
			_id = context['rank']
			for entity in context['entities']:
				if not entity['selected'] and entity['score'] >= reduced_threshold:
					canon_entity = entity['canonical']
					if canon_entity not in canon_entities:
						canon_entities[canon_entity] = {'ann': []}
					canon_entities[canon_entity]['ann'].append([_id, entity['entity'], entity['start'], entity['answer'], entity['score']])
					if _id not in doc_scores or (_id in doc_scores and doc_scores[_id] < entity['score']):
						doc_scores[_id] = entity['score']
					entity['selected'] = True
		# num_entities = sum([entity['selected'] for context in contexts for entity in context['entities']])
		num_canon_entities = sum([int(len(canon_entities[ent]['ann'])>0) for ent in canon_entities])
		if num_canon_entities >= MINIMUM_CANON_ENTITIES:
			break
		reduced_threshold -= 0.1
	reduced_threshold = max(0, reduced_threshold)
	return contexts, canon_entities, doc_scores, round(reduced_threshold, 2)


def get_entities(contexts, predictions, nlp, span_threshold=0.0, **kwargs):
	# choose between different NER methods if required
	return get_entities_spacy(contexts, predictions, nlp, span_threshold)


def get_sentence_boundaries(context_dict, nlp):
	# sentence_boundaries -> dict(cid: list[sent.start, sent.end]) 
	# nlp = spacy.load('en_core_web_sm')
	sentence_boundaries = {}
	for cid in context_dict:
		ann_context = nlp(context_dict[cid])
		boundaries = [(ann_context[sent.start].idx, ann_context[sent.end-1].idx+len(ann_context[sent.end-1])) for sent in ann_context.sents]
		sentence_boundaries[cid] = boundaries
	return sentence_boundaries


def get_sentence(entity, context, sentence_boundaries, answer_start):
	boundary_match = [(start, end) for start, end in sentence_boundaries if answer_start >= start and answer_start < end]
	if len(boundary_match)>0:
		c_start, c_end = boundary_match[0]
	else:
		print('No sent match for entity: %s \nin context:%s.\n Returning whole context.'%(entity, context))
		c_start, c_end = 0, len(context)
	return context[c_start:c_end]


def get_entailment_scores(answer_type, canon_entities, context_dict, sentence_boundaries, entpredictor):
	entailment = {}
	for canon_entity in canon_entities:
		inputs, predictions = [], []
		for cid, entity, start, answer, conf in canon_entities[canon_entity]['ann']:
			context = context_dict[cid]
			sentence = get_sentence(entity, context, sentence_boundaries[cid], int(start))
			hypothesis = entity + ' is a ' + answer_type
			inputs.append({"premise": sentence, "hypothesis": hypothesis, "label": None})
		# print('Num inputs to ent predictor: ', len(inputs))
		if len(inputs) > 0:
			predictions = entpredictor.predict_batch_json(inputs)
		ent_probs = [EntailmentLabelProbs(*prediction['probs']) for prediction in predictions]
		entailment[canon_entity] = {}
		for entity_tuple, prob in zip(canon_entities[canon_entity]['ann'], ent_probs):
			if prob.entailment >= prob.neutral and prob.entailment >= prob.contradiction:
				cid = entity_tuple[0]
				if cid not in entailment[canon_entity]:
					entailment[canon_entity][cid] = prob.entailment
				else:
					if prob.entailment > entailment[canon_entity][cid]:
						entailment[canon_entity][cid] = prob.entailment
	return entailment


def rank_entities(canon_entities, entailment, doc_scores, sentence_boundaries):
	winning_doc = sorted(doc_scores.items(), key=lambda x:[1], reverse=True)[0][0]
	D = len(sentence_boundaries) # count #contexts
	for canon_entity in canon_entities:
		d = len(canon_entities[canon_entity]['ann']) 
		_cids, sum_answer_conf,sum_entailment = [], 0.0, 0.0
		_cids = [cid for cid,_,_,_,_ in canon_entities[canon_entity]['ann']]
		sum_answer_conf = sum([conf for _,_,_,_, conf in canon_entities[canon_entity]['ann']])
		if canon_entity in entailment and len(entailment[canon_entity].items()) > 0:
			sum_entailment = sum([ent for cid, ent in entailment[canon_entity].items()])
		P = sum_answer_conf/float(d) if d>0 else 0.0
		E = sum_entailment/float(d) if d>0 else 0.0
		canon_entities[canon_entity]['type_compatibility_score'] = E
		canon_entities[canon_entity]['answer_confidence_score'] = P
		canon_entities[canon_entity]['context_frequency_score'] = float(d)/D
		canon_entities[canon_entity]['winning_document_score'] = 0 if winning_doc not in _cids else \
			doc_scores[winning_doc]
	entities_flat = []
	for canon_entity in canon_entities:
		ann = []
		for cid, entity, start, answer, conf in canon_entities[canon_entity]['ann']:
			if canon_entity in entailment and cid in entailment[canon_entity]:
				ent = entailment[canon_entity][cid]
			else:
				ent = 0.0
			ann.append((cid, entity, start, answer, conf, ent))
		canon_entities[canon_entity]['ann'] = ann
	return canon_entities


def predict_enumerations(query, qtuples, contexts, qa, nlp, entpredictor, topk=10, span_threshold=0.4):
	## 1. Span prediction
	print('Enum: Getting model predictions')
	tic=time.perf_counter()
	predictions = get_model_predictions(qa, query, contexts, topk)
	toc = time.perf_counter()
	print('Completed in: %.4f secs'%(toc-tic))
	## 2. Get ranked named-entity mentions
	print('Enum: Getting named entities')
	annotated_contexts, canon_entities, doc_scores, reduced_threshold = get_entities(contexts, predictions, nlp, span_threshold)
	tic = time.perf_counter()
	print('Completed in: %.4f secs'%(tic-toc))
	## 3. Rank entities
	print('Enum: Getting ranked entities')
	context_dict = {ann['rank']: ann['context'] for ann in annotated_contexts}
	sentence_boundaries = get_sentence_boundaries(context_dict, nlp)
	entailment = get_entailment_scores(qtuples.type, canon_entities, context_dict, sentence_boundaries, entpredictor)
	canon_entities = rank_entities(canon_entities, entailment, doc_scores, sentence_boundaries)
	toc = time.perf_counter()
	print('Completed in: %.4f secs'%(toc-tic))
	# ## 4. updating entity annotation scores
	# print('Enum: Updating enum scores in annotations')
	# for context in annotated_contexts:
	# 	for ent in context['entities']:
	# 		ent['score'] = canon_entities[ent['canonical']]['type_compatibility_score']
	# tic = time.perf_counter()
	# print('Completed in: %.4f secs'%(tic-toc))
	print('Enum: Returning enum predictions')
	return canon_entities, annotated_contexts, reduced_threshold