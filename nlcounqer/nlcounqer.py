from flask import Flask, render_template, url_for, json, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
from pipeline import pipeline
import json
import pprint
import signal
import sys, os
import glob
from retrieval.bing_search import call_bing_api

try: 
	import urllib2 as myurllib
except ImportError:
	import urllib.request as myurllib


# define paths
cache_path = 'static/data/'
tmp_path = './'


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

## endpoint for 
@app.route('/ftresults', methods=['GET', 'POST'])
@cross_origin()
def free_text_query():
	# query parsing for ajax call
	args = request.args
	query = args['query']
	numsnippets = args['snippets'] 
	
	# check for optional arguments from aggregator calls
	model = args['model'] if 'model' in args else None
	aggregator = args['aggregator'] if 'aggregator' in args else None
	
	print("Query: %s\n#snippets: %s\nmodel: %s\naggregator: %s\n", (query, numsnippets, model, aggregator))
	response = pipeline(query, numsnippets, model, aggregator) if len(query) > 0 else {}
	# pprint.pprint(response, width=160)
	return jsonify(response)

@app.route('/')
@cross_origin()
def display_mainpage():
        #return "Hello World!"
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True, port=5000)