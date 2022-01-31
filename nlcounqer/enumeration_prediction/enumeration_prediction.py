import re
import os
import time

def get_model_predictions(qa, query, contexts, topk):
	## server edit ##
	# model = AutoModelForQuestionAnswering.from_pretrained('mrm8488/spanbert-finetuned-squadv2', cache_dir='/.cache/huggingface/transformers/')
	# tokenizer = AutoTokenizer.from_pretrained('mrm8488/spanbert-finetuned-squadv2', cache_dir='/.cache/huggingface/transformers/')
	# qa = pipeline('question-answering', model=model, tokenizer=tokenizer)
	# qa = pipeline('question-answering', 'mrm8488/spanbert-finetuned-squadv2')
	predictions = {}
	for item in contexts:
		if query.startswith('How many ') or query.startswith('how many '):
			query = 'Which ' + query[len('how many '):]
		try:
			answer = qa(question=query, context=item['context'], topk=topk)
		except:
			answer = []
		finally:
			predictions[item['rank']] = answer
	return predictions


def get_entities_stanford_nlp(contexts, predictions, qtuples, nlp):
	numeric_classes = ['DATE', 'PERCENT', 'TIME', 'MONEY', 'CARDINAL', 'QUANTITY']
	# nlp = stanza.Pipeline('en',dir='/home/shrestha/stanza_resources')
	## server edits ##
	# nlp = stanza.Pipeline('en', dir='/root/stanza_resources')
	entities_per_context = {}
	for context in contexts:
		_id = context['rank']
		entities = {}
		for answer in predictions[_id]:
			# ann = client.annotate(answer['answer'])
			ann = nlp(answer['answer'])
			# mentions = [mention for mention in ann.mentions if mention.entityType not in temporal_numeric_classes]
			mentions = [mention for mention in ann.entities if mention.type not in numeric_classes]
			for mention in mentions:
				entity_startchar = mention.start_char
				entity = { 
						  'score': round(answer['score'], 2), 
						  'start': answer['start'] + entity_startchar,
						  'answer': answer['answer']
						 }
				if mention.text in entities and entity['start'] == entities[mention.text]['start']:
					## keep the highest scoring alternative for same entity mentions
					if entity['score'] > entities[mention.text]['score']:
						entities[mention.text]['score'] = entity['score']
				else:
					entities[mention.text] = entity
		entities_per_context[_id] = entities
		context['entities'] = entities
	return entities_per_context, contexts


def get_entities_spacy(contexts, predictions, qtuples, nlp):
	numeric_classes = ['DATE', 'PERCENT', 'TIME', 'MONEY', 'CARDINAL', 'QUANTITY', 'ORDINAL']
	# nlp = spacy.load("en_core_web_sm")
	entity_to_cid_dict = {}
	entities_per_context = {}
	for context in contexts:
		_id = context['rank']
		entities = {}
		for answer in predictions[_id]:
			ann = nlp(answer['answer'])
			mentions = [mention for mention in ann.ents if mention.label_ not in numeric_classes]
			for mention in mentions:
				entity_startchar = mention.start_char
				entity = { 
						  'score': round(answer['score'],2), 
						  'start': answer['start'] + entity_startchar,
						  'answer': answer['answer'],
						  'entity': mention.text
						 }
				if mention.text in entities and entity['start'] == entities[mention.text]['start']:
					if entity['score'] > entities[mention.text]['score']:
						entities[mention.text]['score'] = entity['score']
				else:
					entities[mention.text] = entity
				if mention.text not in entity_to_cid_dict:
					entity_to_cid_dict[mention.text] = []
				if _id not in entity_to_cid_dict[mention.text]:
					entity_to_cid_dict[mention.text].append(_id)
		entities_per_context[_id] = entities
		context['entities'] = sorted([v for k, v in entities.items()], key=lambda x:x['start'])
		# print('entities: ', context['entities'])
	return entities_per_context, contexts, entity_to_cid_dict


def rank_entities(entities_by_id, entity_to_cid_dict):
	entities_flat = [(_id, entity, entities[entity]['score'], entities[entity]['start']) for _id, entities in entities_by_id.items() for entity in entities]		
	D = set([_id for _id, _, _, _ in entities_flat])
	entities_flat = [(_id, entity, conf, start, len(entity_to_cid_dict[entity])/len(D)) for _id, entity, conf, start in entities_flat]
	entities_sorted = sorted(entities_flat, key=lambda x: x[4], reverse=True)
	return entities_sorted


def get_entities(contexts, predictions, qtuples, nlp, ner='spacy'):
	if ner == 'spacy':
		return get_entities_spacy(contexts, predictions, qtuples, nlp)
	else:
		return get_entities_stanford_nlp(contexts, predictions, qtuples, nlp)


def predict_enumerations(query, qtuples, contexts, qa, nlp, ner='spacy', topk=10):
	## 1. Span prediction
	print('Enum: Getting model predictions')
	tic=time.perf_counter()
	predictions = get_model_predictions(qa, query, contexts, topk)
	toc = time.perf_counter()
	print('Completed in: %.4f secs'%(toc-tic))
	## 2. Get ranked named-entity mentions
	print('Enum: Getting named entities')
	entities_by_id, annotated_contexts, entity_to_cid_dict = get_entities(contexts, predictions, qtuples, nlp, ner)
	tic = time.perf_counter()
	print('Completed in: %.4f secs'%(tic-toc))
	## 3. Rank entities
	print('Enum: Getting ranked entities')
	ranked_entities = rank_entities(entities_by_id, entity_to_cid_dict)
	toc = time.perf_counter()
	print('Completed in: %.4f secs'%(toc-tic))
	print('Enum: Returning enum predictions')
	return ranked_entities, annotated_contexts