from transformers import pipeline
from stanza.server import CoreNLPClient
import re


def get_model_predictions(query, contexts, topk):
	model = pipeline('question-answering', 'mrm8488/spanbert-finetuned-squadv2')
	predictions = {}
	for item in (contexts):
		if query.startswith('How many ') or query.startswith('how many '):
			query = 'Which ' + query[len('how many '):]
		try:
			answer = model(question=query, context=item['context'], topk=topk)
		except:
			answer = []
		finally:
			predictions[item['rank']] = answer
	return predictions


def get_entities(predictions, qtuples, client_parameters):
	temporal_numeric_classes = ['DATE', 'PERCENT', 'TIME', 'MONEY']

	# with open(entityfile, 'w') as fp:
	# 	fieldnames = ['qid','cid','query','answer_type','qentity','qrelation','qcontext','context','answer','entity','start_char','score']
	# 	writer = csv.DictWriter(fp, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
	# 	writer.writeheader()

	with CoreNLPClient(annotators=client_parameters['annotators'], 
					   timeout=client_parameters['timeout'], 
					   memory=client_parameters['memory'],
					   properties=client_parameters['properties']) as client:
		entities_per_context = {}
		for id in predictions:
			entities = {}
			for answer in predictions[id]:
				ann = client.annotate(answer['answer'])
				mentions = [mention for mention in ann.mentions if mention.entityType not in temporal_numeric_classes]
				for mention in mentions:
					sentenceTokenOffsetBegin = ann.sentence[mention.sentenceIndex].tokenOffsetBegin
					entity_startchar = ann.sentence[mention.sentenceIndex].token[mention.tokenStartInSentenceInclusive - sentenceTokenOffsetBegin].beginChar
					entity = { 
							  'score': answer['score'], 
							  'start': answer['start'] + entity_startchar,
							  'answer': answer['answer']
							 }
					if mention.entityMentionText in entities and entity['start'] == entities[mention.entityMentionText]['start']:
						## keep the highest scoring alternative for same entity mentions
						if entity['score'] > entities[mention.entityMentionText]['score']:
							entities[mention.entityMentionText]['score'] = entity['score']
					else:
						entities[mention.entityMentionText] = entity
			entities_per_context[id] = entities
		return entities_per_context
			

def predict_enumerations(query, qtuples, contexts, topk=10):
	client_parameters = {
		'annotators': ['tokenize', 'ssplit', 'pos', 'lemma', 'depparse', 'ner'],
		'timeout': 30000,
		'memory': '2G',
		'properties': {
			'ner.applyNumericClassifiers': 'false', 
			'ner.useSUTime': 'false'
		}
	}
	## 1. Span prediction
	predictions = get_model_predictions(query, contexts, topk)
	## 2. Get ranked named-entity mentions
	entities = get_entities(predictions, qtuples, client_parameters)
	## 3. Rank entities
	