import spacy

from spacy.tokens import Span, Token
from tqdm import tqdm
from inflection import singularize
from collections import namedtuple


class Entity(object):
	def __init__(self, span: Span):
		self.span = span

	def __eq__(self, other):
		return (isinstance(other, self.__class__) and 
				other.span.text == self.span.text and
				other.span.start == self.span.start and
				other.span.end == self.span.end)

	def __hash__(self):
		return hash(self.span.text + str(self.span.start) + str(self.span.end))

	def substrOf(self, token) -> bool:
		if type(token) == Token:
			return self.span.text == token.text and self.span.start == token.i and len(self.span) == 1
		elif type(token) == Span:
			return self.span.text in token.text and self.span.start >= token.start and self.span.end <= token.end
		else:
			return False


class ContextToken(object):
	def __init__(self, token: Token):
		self.token = token

	def __eq__(self, other):
		return (isinstance(other, self.__class__) and
				other.token.text == self.token.text and 
				other.token.i == self.token.i)

	def __hash__(self):
		return hash(str(self.token.i) + self.token.text)


class QueryModel(object):
	def __init__(self, query: str, model: str):
		self.query = query
		self.nlp = model

	def query_processing(self):
		entity_classes = ['person', 'norp', 'fac', 'org', 'gpe', 'loc', 'product', 'event', 'work_of_art', 'law', 'language']
		query_annotated = self.nlp(self.query)

		## gathernig entities
		## from NER
		entities = set([Entity(entity) for entity in query_annotated.ents if entity.label_.lower() in entity_classes])
		## from consecutive PROPN POS tags
		start = end = None
		for token in query_annotated:
			if token.pos_.lower() == 'propn' and start is None:
				start = token.i
			elif token.pos_.lower() != 'propn' and start is not None:
				end = token.i
				entities.add(Entity(query_annotated[start:end]))
				start = end = None
		## if propn is at the end of sentence
		if start is not None:
			entities.add(Entity(query_annotated[start:start+1]))

		answer_type = None
		relation = None
		context = set()
		non_contextual_pos = ['adp', 'aux', 'conj', 'det', 'pron', 'punct', 'sconj']
		
		entity_tokens = set([ContextToken(token) for entity in entities for token in entity.span])
		for token in query_annotated:
			if token.pos_.lower() == 'noun' and answer_type == None: ## first noun is the answer type ### heuristic
				start_idx = token.i
				for child in token.children:
					if child.pos_.lower() in ['noun', 'adj'] and child.text.lower() != 'many' and child.i < start_idx:
						start_idx = child.i
				answer_type = query_annotated[start_idx: token.i+1]

			elif token.pos_.lower() == 'noun':
				if type(answer_type) == Token:
					start_idx = answer_type.i
					end_idx = end_idx + 1
				else:
					start_idx = answer_type.start
					end_idx = answer_type.end
				if token.i == end_idx:
					answer_type = query_annotated[start_idx:end_idx+1]
				else:
					if token.text.lower() not in ['how', 'many'] and ContextToken(token) not in entity_tokens and token.pos_.lower() not in non_contextual_pos:
						context.add(ContextToken(token))
		
			elif token.pos_.lower() == 'verb' and token.dep_ == 'ROOT': ## if the root word is a verb then it is the relation 
				relation = token
			elif token.pos_.lower() == 'verb' and relation == None:
				relation = token
			elif token.text.lower() not in ['how', 'many'] and ContextToken(token) not in entity_tokens and token.pos_.lower() not in non_contextual_pos:
				context.add(ContextToken(token))

		## remove entity mentions which form a substring of the answer_type
		entities = [entity for entity in entities if not entity.substrOf(answer_type)]

		if answer_type is not None:
			if type(answer_type) == Span:
				answer_type_tokens = set([ContextToken(token) for token in answer_type])
			else:
				answer_type_tokens = set([ContextToken(answer_type)])
			context = context - answer_type_tokens
		if relation is not None:
			context = context - set([ContextToken(relation)])

		self.type = '' if answer_type is None else singularize(answer_type.text)
		self.entity = tuple(entity.span.text for entity in entities)
		self.relation = '' if relation is None else relation.text
		self.context = tuple(word.token.text for word in context) 


def get_qtuples(query, model):
	query_model = QueryModel(query, model)
	query_model.query_processing()
	QTuples = namedtuple('QTuples', 'type entity relation context')
	qtuples = QTuples(type=query_model.type, 
					  entity=query_model.entity,
					  relation=query_model.relation,
					  context=query_model.context)
	return(qtuples)