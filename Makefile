run:

	pip install -r requirements.txt && python3 ./nlp_tfidf/tfidf.py

build:
	python3 setup.py sdist build && python3 setup.py bdist_wheel --universal && python3 setup.py sdist bdist_wheel

release:
	twine upload dist/*
