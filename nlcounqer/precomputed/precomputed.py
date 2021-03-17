import pandas as pd
from precomputed.precomputed_query import precomputed_query_type, precomputed_query_id
import ast
import json

file_maps = {
	'query': {
		'coquad': '//nlcounqer/static/data/queries/coquad_v1/test_ntuples.csv',
		'stresstest': '//nlcounqer/static/data/queries/stresstest_v1/stresstest_ntuples.csv'
	},
	'snippets_bing': {
		'coquad': '//nlcounqer/static/data/snippets/coquad_v1/test_v1.json',
		'stresstest': '//nlcounqer/static/data/snippets/stresstest_v1/stresstest_v1.json'
	},
	'snippets_wikip': {
		'coquad': '//nlcounqer/static/data/snippets/coquad_v1/test_dpr_retrieed_top50_v1.json',
		'stresstest': '//nlcounqer/static/data/snippets/stresstest_v1/stresstest_dpr_retrieed_top50_v1.json'
	}
}

def get_qtuples(qtype, qid):
	query_df = pd.read_csv(file_maps['query'][qtype])
	tuples = query_df.loc[11, ['answer_type','query_entity','relation','context']].to_dict()
	if not pd.isnull(tuples['context']):
		qtuples['context'] = ast.literal_eval(tuples['context'])
		qtuples['context'] = ';'.join(qtuples['context'])
	else:
		qtuples['context'] = ''
	if not pd.isnull(tuples['relation']):
		qtuples['relation'] = tuples['relation']
	else:
		qtuples['relation'] = ''
	if not pd.isnull(tuples['answer_type']):
		qtuples['type'] = tuples['answer_type']
	else:
		qtuples['type'] = ''
	if not ps.isnull(tuples['query_entity']):
		qtuples['entity'] = ast.literal_eval(tuples['query_entity'])
		qtuples['entity'] = ';'.join(qtuples['entity'])
	return qtuples


def get_contexts(qtype, qid):
	contexts = []
	with open(file_maps['snippets_bing'][qtype]) as fp:
		data = json.load(fp)
		for para in data['data'][0]['paragraphs']:
			_qid = int(para['qas'][0]['id'].split('_')[0])
			cid = int(para['qas'][0]['id'].split('_')[1])
			if qid != qid:
				continue
			contexts.append({'context': para['context'], 'rank': cid})
	return contexts

	
def precomputed_queries(query, tfmodel, thresholds, aggregator):
	result = {}
	qtype = precomputed_query_type(query)
	qid = precomputed_query_id(query)
	result['qtuples'] = get_qtuples(qtype, qid)
	contexts = get_contexts(qtype, qid)
	return result