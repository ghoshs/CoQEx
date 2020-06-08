from flask import Flask, render_template, url_for, json, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
# from get_count_data import related_predicate
from free_text_search import text_tags
import spacy
from spacy.tokens import DocBin
import json
import pprint

from bing_search.bing_search import call_bing_api

try: 
	import urllib2 as myurllib
except ImportError:
	import urllib.request as myurllib


# define cache path
cache_path = 'static/data/'

# setup BERT server 
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
## server edit
# model_dir = '/root/main/bert_model/cased_L-12_H-768_A-12/'
model_dir = '/home/shrestha/Documents/PhD/BERT_models/cased_L-12_H-768_A-12/'


args = get_args_parser().parse_args(['-model_dir', model_dir,
                                     '-port', '5555',
                                     '-port_out', '5556',
                                     '-max_seq_len', 'NONE',
                                     '-mask_cls_sep',
                                     '-cpu'])
server = BertServer(args)


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
	
	print("Query:: ", query)
	if snippetfile is None or model is None:
		response = text_tags(query, numsnippets) if len(query) > 0 else {}
	else:
		with open(cache_path+snippetfile) as fp:
			all_snippets = json.load(fp)
		snippets = all_snippets[query]
		response = text_tags(query, numsnippets, model, snippets)
	# pprint.pprint(response, width=160)
	return jsonify(response)

@app.route('/')
@cross_origin()
def display_mainpage():
        #return "Hello World!"
	return render_template('index.html')

if __name__ == '__main__':
	# start BERT server
	server.start()
	## server edit ##
    # app.run(debug=True)
	app.run(debug=True, port=5000)

	# server.close()