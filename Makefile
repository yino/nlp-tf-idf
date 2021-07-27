run:
	python ./nlp_tfidf/tfidf.py

build:
	python setup.py sdist build && python setup.py bdist_wheel --universal && python3 setup.py sdist bdist_wheel

release:
	twine upload dist/*
