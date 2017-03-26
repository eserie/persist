pep8:
	flake8

tests:
	py.test -v persist/

coverage:
	py.test --cov-report=html --cov-report=term --pdb --cov persist -v persist/

