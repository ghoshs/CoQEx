from flask import Flask, render_template, url_for, json, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import json
import pprint
import signal
import sys, os
import glob
import traceback
import spacy
import configparser
os.environ['TRANSFORMERS_CACHE'] = '/.cache/huggingface/transformers/'
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from sentence_transformers import SentenceTransformer
# Download model beforehand -> set proxies on the terminal beforehand (export http-proxy.. )
# load model from_pretrained with cache_dir passed
from pipeline import pipeline as nlcounqer_pipeline
from retrieval.bing_search import call_bing_api
from precomputed.query import is_precomputed, query_list
from precomputed.precomputed import precomputed_queries

try: 
	import urllib2 as myurllib
except ImportError:
	import urllib.request as myurllib

model=tfmodel=thresholds=qa_enum=nlp=None

proxies = {




}

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
def load_models(model):
	### count models
	count_config = configparser.ConfigParser()
	# count_config.read('//nlcounqer/count_prediction/count_config.ini')
	## server edit ##
	count_config.read('/nlcounqer/count_prediction/count_config_server.ini')
	
	model_path_dict = json.load(open(count_config['paths']['ModelPath'], 'r'))
	model_path = model_path_dict[model]['model_path']
	thresholds = model_path_dict[model]['thresholds']
	qa_count = pipeline("question-answering", model_path)
	
	### enum models
	enum_config = configparser.ConfigParser()
	# enum_config.read('//nlcounqer/enumeration_prediction/enum_config.ini')
	## server edit ##
	enum_config.read('/nlcounqer/enumeration_prediction/enum_config_server.ini')

	nlp = spacy.load(enum_config['nlp']['Language'])
	tokenizer = AutoTokenizer.from_pretrained(enum_config['paths']['Model'], cache_dir=enum_config['paths']['CacheDir'], local_files_only=True)
	model = AutoModelForQuestionAnswering.from_pretrained(enum_config['paths']['Model'], cache_dir=enum_config['paths']['CacheDir'])
	qa_enum = pipeline('question-answering', model=model, tokenizer=tokenizer)
	# qa_enum = pipeline('question-answering', enum_config['paths']['Model'], cache_dir=enum_config['paths']['CacheDir'])
	
	## load sbert for count contextualization
	sbert = SentenceTransformer(count_config['sbert']['SentBERTModel']) 
	return qa_count, thresholds, qa_enum, nlp, sbert
	

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
	global model, tfmodel, thresholds, qa_enum, nlp
	# query parsing for ajax call
	args = request.args
	query = args['query']
	numsnippets = args['snippets'] 
	
	# check for optional arguments from aggregator calls
	args_model = args['model'] if 'model' in args else 'default'
	aggregator = args['aggregator'] if 'aggregator' in args else 'weighted'
	staticquery = args['staticquery'] if 'staticquery' in args else 'live'
	if not model or model != args_model:
		model = args_model
		tfmodel, thresholds, qa_enum, nlp, sbert = load_models(model)

	print("Query: %s\n#snippets: %s\nmodel: %s\naggregator: %s\n"%(query, numsnippets, model, aggregator))
	if staticquery == 'precomputed' and is_precomputed(query):
		print('precomputed!!')
		# return response and time elapsed in seconds
		if len(query) > 0:
			response, time_elapsed = precomputed_queries(query, tfmodel, thresholds, aggregator)
		else:
			response, time_elapsed = {}, 0.0
	else:
		print('Querying live!!')
		try:
			response, time_elapsed = nlcounqer_pipeline(query, tfmodel, thresholds, qa_enum, nlp, sbert, aggregator, numsnippets) if len(query) > 0 else {}
		except Exception:
			print(traceback.format_exc())
			response, time_elapsed = {}, 0.0
	response['q'] = query
	response['time_in_sec'] = time_elapsed 
	# pprint.pprint(response, width=160)
	return jsonify(response)

@app.route('/')
@cross_origin()
def display_mainpage():
        #return "Hello World!"
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True, port=5000)