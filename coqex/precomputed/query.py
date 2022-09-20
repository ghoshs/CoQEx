import pandas as pd
import os
path = '/coqex/'

file_maps = {
	'query': {
		# 'coquad_100': path + 'static/data/queries/coquad_100_v1/coquad_100_ntuples.csv',
		'kg50': path + 'static/data/queries/coquad_v1/kg50_ntuples.csv',
		'snippet100': path + 'static/data/queries/coquad_v1/snippet100_ntuples.csv',
		'organic100': path + 'static/data/queries/coquad_v1/organic100_ntuples.csv',
		# 'stresstest': path + '/static/data/queries/stresstest_v1/stresstest_ntuples.csv'
	},
	'snippets_bing': {
		'coquad_100': path + 'static/data/snippets/coquad_v1/test_v1.json',
		'kg50': path + 'static/data/snippets/coquad_v1/kg50.json',
		'snippet100': path + 'static/data/snippets/coquad_v1/snippet100.json',
		'organic100': path + 'static/data/snippets/coquad_v1/organic100.json',
		# 'stresstest': path + 'static/data/snippets/stresstest_v1/stresstest_v1.json'
	},
	# 'snippets_wikip': {
	# 	'coquad': path + 'static/data/snippets/coquad_v1/test_dpr_retrieed_top50_v1.json',
	# 	'stresstest': path + 'static/data/snippets/stresstest_v1/stresstest_dpr_retrieed_top50_v1.json'
	# },
	'counts_inverse': {
		'coquad_100': path + 'static/data/count_info/coquad_v1/bing/weighted_analysis_0.5.csv',
		# 'coquad': path + 'static/data/count_info/coquad_v1/bing/weighted_analysis_0.5.csv',
		'stresstest': path + 'static/data/count_info/stresstest_v1/bing/weighted_0.0_boosted_inverse.csv'
	},
	'enums_inverse': {
		'coquad_100': path + 'static/data/count_info/coquad_100_v1/bing/entities_strength_ranker_ranked.csv',
		# 'coquad': path + 'static/data/count_info/coquad_v1/bing/reranked_binary.csv',
		'stresstest': path + 'static/data/count_info/stresstest_v1/bing/reranked_binary.csv'
	}
}

def is_coquad_100_query(query):
	q1 = pd.read_csv(path+'static/data/queries/coquad_100_v1/coquad_100_ntuples.csv')
	if query.lower() in [q.lower() for q in q1['query'].values]:
		return True
	else:
		return False


def is_coquad_query(query):
	q1 = pd.read_csv(path+'static/data/queries/coquad_v1/test_ntuples.csv')
	if query.lower() in [q.lower() for q in q1['query'].values]:
		return True
	else:
		return False


def is_stresstest_query(query):
	q1 = pd.read_csv(path+'static/data/queries/stresstest_v1/stresstest_ntuples.csv')
	if query.lower() in [q.lower() for q in q1['query'].values]:
		return True
	else:
		return False

def is_kg50_query(query):
	q1 = pd.read_csv(path+'static/data/queries/coquad_v1/kg50_ntuples.csv')
	if query.lower() in [q.lower() for q in q1['query'].values]:
		return True
	else:
		return False


def is_snippet100_query(query):
	q1 = pd.read_csv(path+'static/data/queries/coquad_v1/snippet100_ntuples.csv')
	if query.lower() in [q.lower() for q in q1['query'].values]:
		return True
	else:
		return False


def is_organic100_query(query):
	q1 = pd.read_csv(path+'static/data/queries/coquad_v1/organic100_ntuples.csv')
	if query.lower() in [q.lower() for q in q1['query'].values]:
		return True
	else:
		return False


def is_precomputed(query):
	# if is_coquad_100_query(query):
		# return True
	if is_kg50_query(query):
		return True
	elif is_snippet100_query(query):
		return True
	elif is_organic100_query(query):
		return True
	# elif is_stresstest_query(query):
	# 	return True
	else:
		return False


def precomputed_query_type(query):
	# if is_coquad_100_query(query):
		# return 'coquad_100'
	if is_kg50_query(query):
		return 'kg50'
	if is_snippet100_query(query):
		return 'snippet100'
	if is_organic100_query(query):
		return 'organic100'
	# elif is_stresstest_query(query):
		# return 'stresstest'
	# else:
	return 'invalid'


def precomputed_query_id(query):
	if precomputed_query_type(query) == 'coquad':
		q1 = pd.read_csv(path+'static/data/queries/coquad_v1/test_ntuples.csv')
	elif precomputed_query_type(query) == 'stresstest':
		q1 = pd.read_csv(path+'static/data/queries/stresstest_v1/stresstest_ntuples.csv')
	elif precomputed_query_type(query) == 'coquad_100':
		q1 = pd.read_csv(path+'static/data/queries/coquad_100_v1/coquad_100_ntuples.csv')
	elif precomputed_query_type(query) == 'kg50':
		q1 = pd.read_csv(path+'static/data/queries/coquad_v1/kg50_ntuples.csv')
	elif precomputed_query_type(query) == 'snippet100':
		q1 = pd.read_csv(path+'static/data/queries/coquad_v1/snippet100_ntuples.csv')
	elif precomputed_query_type(query) == 'organic100':
		q1 = pd.read_csv(path+'static/data/queries/coquad_v1/organic100_ntuples.csv')
	else:
		print('Query type: ', precomputed_query_type(query))
		raise ValueError('Cannot determine precomputed query type of: \n %s'%query)
	return q1.loc[q1['query'].str.lower() == query.lower(), ['qid']].values[0][0]


def query_list():
	queries = {}
	for query_set in file_maps['query']:
		queries[query_set] = list(pd.read_csv(os.path.join(path,file_maps['query'][query_set]))['query'].values)
	# coquad_queries = list(pd.read_csv(path+'static/data/queries/coquad_v1/test_ntuples.csv')['query'].values)
	# stresstest_queries = list(pd.read_csv(path+'static/data/queries/stresstest_v1/stresstest_ntuples.csv')['query'].values)
	return queries