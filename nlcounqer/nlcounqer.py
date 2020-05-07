from flask import Flask, render_template, url_for, json, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
# from get_count_data import related_predicate
from free_text_search import text_tags
import spacy
from spacy.tokens import DocBin
import json
import pprint

try: 
	import urllib2 as myurllib
except ImportError:
	import urllib.request as myurllib

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/ftresults', methods=['GET', 'POST'])
@cross_origin()
def free_text_query():
	## query parsing for displacy code
	# query = json.loads(request.data.decode())['text']
	# query parsing for ajax call
	query = request.args.get('query')
	snippets = request.args.get('snippets')
	print("Query:: ", query)
	response = text_tags(query, snippets) if len(query) > 0 else {}
	# pprint.pprint(response, width=160)
	return jsonify(response)

@app.route('/')
@cross_origin()
def display_mainpage():
        #return "Hello World!"
	return render_template('index.html')

if __name__ == '__main__':
        #app.run()
	app.run(debug=True, port=5000)