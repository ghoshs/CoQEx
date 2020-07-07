from flask import Flask, render_template, url_for, json, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
# from get_count_data import related_predicate
from free_text_search import text_tags
import spacy
from spacy.tokens import DocBin
import json
import pprint
import signal
import sys, os
import glob

from bing_search.bing_search import call_bing_api

try: 
	import urllib2 as myurllib
except ImportError:
	import urllib.request as myurllib


# define paths
cache_path = 'static/data/'
tmp_path = './'

# setup BERT server 
from bert_serving.server.helper import get_args_parser, get_shutdown_parser
from bert_serving.server import BertServer
## server edit
model_dir = '/root/main/bert_model/cased_L-12_H-768_A-12/'
# model_dir = '/home/shrestha/Documents/PhD/BERT_models/cased_L-12_H-768_A-12/'
server = None

args = get_args_parser().parse_args(['-model_dir', model_dir,
                                     '-port', '5555',
                                     '-port_out', '5556',
                                     '-max_seq_len', 'NONE',
                                     '-mask_cls_sep',
                                     '-cpu'])
shut_args = get_shutdown_parser().parse_args(['-ip','localhost','-port','5555','-timeout','5000'])

# BERT server setup
def setupBERT():
	if server is None:
		return BertServer(args)
	else:
		return server

# app graceful shutdown
def signal_handler(signal, frame):
	# shutdown bert server
	server.shutdown(shut_args)
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

signal.signal(signal.SIGINT, signal_handler)

# flask app config
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/snippets', methods=['GET', 'POST'])
@cross_origin()
def get_snippets():
	query = request.args.get('query')
	snippets = request.args.get('snippets')
	print("Query:: ", query)
	response = call_bing_api(query, snippets)
	# pprint.pprint(response, width=160)
	return jsonify(response)

@app.route('/ftresults', methods=['GET', 'POST'])
@cross_origin()
def free_text_query():
	## query parsing for displacy code
	# query = json.loads(request.data.decode())['text']
	# query parsing for ajax call
	args = request.args
	query = args['query']
	numsnippets = args['snippets'] 
	
	# check for optional arguments from aggregator calls
	snippetfile = args['snippetcache'] if 'snippetcache' in args else None
	model = args['model'] if 'model' in args else None
	threshold = float(args['threshold']) if 'threshold' in args else None
	config = float(args['config']) if 'config' in args else None
	
	print("Query: ", query, " #snippets: ", numsnippets, " Model: ", model, " Threshold: ", threshold, " Config: ", config)
	if snippetfile is None:
		response = text_tags(query, numsnippets, model, config, threshold) if len(query) > 0 else {}
	else:
		with open(cache_path+snippetfile) as fp:
			all_snippets = json.load(fp)
		snippets = all_snippets[query]
		response = text_tags(query, numsnippets, model, config, threshold, snippets)
	# pprint.pprint(response, width=160)
	return jsonify(response)

@app.route('/')
@cross_origin()
def display_mainpage():
        #return "Hello World!"
	return render_template('index.html')

if __name__ == '__main__':
	# start BERT server
	# server = setupBERT()
	# server.start()
	## server edit ##
    # app.run(debug=True)
	app.run(debug=True, port=5000)