setup-env:
	conda env create --file=environment.yml


write-cv-index:
	python cli/write_cv_index.py