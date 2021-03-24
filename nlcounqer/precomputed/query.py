import pandas as pd

path = '//nlcounqer/'
## server edit ##
# path = '/nlcounqer/'

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


def is_precomputed(query):
	if is_coquad_query(query) or is_stresstest_query(query):
		return True
	else:
		return False

def precomputed_query_type(query):
	if is_coquad_query(query):
		return 'coquad'
	else:
		return 'stresstest'


def precomputed_query_id(query):
	if precomputed_query_type(query) == 'coquad':
		q1 = pd.read_csv(path+'static/data/queries/coquad_v1/test_ntuples.csv')
	else:
		q1 = pd.read_csv(path+'static/data/queries/stresstest_v1/stresstest_ntuples.csv')
	return q1.loc[q1['query'].str.lower() == query.lower(), ['qid']].values[0][0]

def query_list():
	coquad_queries = list(pd.read_csv(path+'static/data/queries/coquad_v1/test_ntuples.csv')['query'].values)
	stresstest_queries = list(pd.read_csv(path+'static/data/queries/stresstest_v1/stresstest_ntuples.csv')['query'].values)
	return coquad_queries+stresstest_queries