import requests

# ## server edits - 2 ##

# ## server edits ##



def call_bing_api(query, count=10, subscription_key='YOUR_API_KEY'):
	# global num_api_calls
	url = "https://api.cognitive.microsoft.com/bingcustomsearch/v7.0/search"
	headers = {'Ocp-Apim-Subscription-Key': subscription_key}
	params = {"q": query, "customconfig": "YOUR_CUSTOM_CONFIG", "mkt": "en-US", "safesearch": "Moderate", "responseFilter": "webPages", "count": count}
	
	response = requests.get(url, headers=headers, params=params)
	## server edits ##

	
	response.raise_for_status()
	# num_api_calls += 1
	results = response.json()
	snippets = []
	if 'webPages' in results:
		for rank, item in enumerate(results['webPages']['value']):
			webpage = {}
			webpage['rank'] = rank
			webpage['url'] = item['url'] if 'url' in item else ''
			webpage['about'] = item['about'] if 'about' in item else ''
			webpage['context'] = item['snippet'] if 'snippet' in item else ''
			webpage['dateLastCrawled'] = item['dateLastCrawled'] if 'dateLastCrawled' in item else ''
			snippets.append(webpage)
	return snippets
