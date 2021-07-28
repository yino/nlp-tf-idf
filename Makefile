run:
	pip install -r requirements.txt && python3 ./nlp_tfidf/tfidf.py

release:
	python3 setup.py upload
