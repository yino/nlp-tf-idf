run:
	pip install -r requirements.txt && python3 ./nlp_tfidf/tfidf.py

release:
	python3 setup.py upload

pipy:
	python3 setup.py sdist bdist_wheel && twine upload --repository-url https://upload.pypi.org/legacy/ dist/* -u __token__

