from flask import Flask, render_template, url_for, json, request, jsonify
from flask_cors import CORS, cross_origin
import json
import pprint
import signal
import sys, os
import glob
import traceback
import spacy
import configparser
## set cache directories before loading the predictor module
os.environ['TRANSFORMERS_CACHE'] = '/.cache/huggingface/transformers/'

from transformers import pipeline
from sentence_transformers import SentenceTransformer
## Download model beforehand -> set proxies on the terminal beforehand (export http-proxy.. )
## load model from_pretrained with cache_dir passed
from pipeline import pipeline as nlcounqer_pipeline
from retrieval.bing_search import call_bing_api
from precomputed.query import is_precomputed, query_list
from precomputed.precomputed import prefetched_contexts, get_precomputed_result

## to-do: implement max live queries = 100 per / day; maybe display counter on the website

try: 
	import urllib2 as myurllib
except ImportError:
	import urllib.request as myurllib

model=tfmodel=count_threshold=qa_enum=enum_threshold=typepredictor=nlp=sbert=None
NUM_SNIPPETS = 50
proxies = {}




# app graceful shutdown
def signal_handler(signal, frame):
	# remove tmp files
	files = glob.glob(tmp_path+'tmp*/*', recursive=True)
	folders = ['/'.join(file.split('/')[:-1]) for file in files]
	for file in files:
		try:
			os.remove(file)
		except OSError:
			print('Cannot delete file: ', file)
	for folder in folders:
		try:
			os.rmdir(folder)
		except OSError:
			print('Cannot delete folder: ', folder)
	# shut down server
	sys.exit(0)

# signal.signal(signal.SIGINT, signal_handler)
def load_models(model='default'):
	### count models
	count_config = configparser.ConfigParser()
	count_config.read('/coqex/count_prediction/count_config_server.ini')
	
	model_path_dict = json.load(open(count_config['paths']['ModelPath'], 'r'))
	model_path = model_path_dict[model]['model_path']
	count_threshold = model_path_dict[model]['threshold']
	qa_count = pipeline("question-answering", model_path)
	
	### enum models
	enum_config = configparser.ConfigParser()
	enum_config.read('/coqex/enumeration_prediction/enum_config_server.ini')

	nlp = spacy.load(enum_config['nlp']['Language'])
	model_path = enum_config['paths']['model']
	qa_enum = pipeline('question-answering', model_path)
	enum_threshold = float(enum_config['span']['threshold'])

	typepredictor = pipeline('text-classification', model=enum_config['typeprediction']['model'])

	## load sbert for count contextualization
	sbert = SentenceTransformer(count_config['sbert']['SentBERTModel']) 

	return qa_count, count_threshold, qa_enum, enum_threshold, typepredictor, nlp, sbert
	

# flask app config
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

## Endpoint for accessing snippets for rmeote computations
@app.route('/snippets', methods=['GET', 'POST'])
@cross_origin()
def get_snippets():
	query = request.args.get('query')
	snippets = request.args.get('snippets')
	print("Query:: ", query)
	response = call_bing_api(query, snippets)
	# pprint.pprint(response, width=160)
	return jsonify(response)


## Endpoint for retrieving precomputed query list
@app.route('/get_query_list', methods=['GET', 'POST'])
@cross_origin()
def get_query_list():
	return jsonify(query_list())


## endpoint for 
@app.route('/ftresults', methods=['GET', 'POST'])
@cross_origin()
def free_text_query():
	print('Sucess!! Request:', request)
	print('Computation will start now .. ')
	global model, tfmodel, count_threshold, qa_enum, enum_threshold, typepredictor, nlp, sbert
	# query parsing for ajax call
	args = request.args
	query = args['query']
	
	# check for optional arguments from aggregator calls
	numsnippets = args['snippets'] if 'snippets' in args else NUM_SNIPPETS
	staticquery = args['staticquery'] if 'staticquery' in args else 'live'
	args_model = args['model'] if 'model' in args else 'default'
	aggregator = args['aggregator'] if 'aggregator' in args else 'weighted'
	if not model or model != args_model or not sbert:
		model = args_model
		tfmodel, count_threshold, qa_enum, enum_threshold, typepredictor, nlp, sbert = load_models(model)

	# ####### REMOVE in final version 
	# staticquery = 'prefetched' if staticquery == 'precomputed' else staticquery 

	print("Query: %s\n#snippets: %s\nmodel: %s\naggregator: %s\n"%(query, numsnippets, model, aggregator))
	if staticquery == 'prefetched' and is_precomputed(query):
		print('prefetched!!')
		# return response and time elapsed in seconds
		if len(query) > 0:
			contexts, qtuples = prefetched_contexts(query)
			response, time_elapsed = nlcounqer_pipeline(query, tfmodel, count_threshold, qa_enum, enum_threshold, typepredictor, nlp, sbert, aggregator, numsnippets, contexts=contexts, qtuples=qtuples)
		else:
			response, time_elapsed = {}, 0.0
	elif staticquery == 'precomputed' and is_precomputed(query):
		print('precomputed!!')
		# return response and time elapsed in seconds
		if len(query) > 0:
			response, time_elapsed = get_precomputed_result(query)
	else:
		print('Querying live!!')
		response, time_elapsed = {}, 0.0
		if len(query) > 0:
			try:
				response, time_elapsed = nlcounqer_pipeline(query, tfmodel, count_threshold, qa_enum, enum_threshold, typepredictor, nlp, sbert, aggregator, numsnippets)
			except Exception:
				print(traceback.format_exc())
	response['q'] = query
	response['time_in_sec'] = round(time_elapsed,2) 
	# save_to_cache(response)
	return jsonify(response)

@app.route('/')
@cross_origin()
def display_mainpage():
	global model, tfmodel, count_threshold, qa_enum, enum_threshold, typepredictor, nlp, sbert
	if not model or not sbert:
		model = "default"
		tfmodel, count_threshold, qa_enum, enum_threshold, typepredictor, nlp, sbert = load_models(model)
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True, port=5000)