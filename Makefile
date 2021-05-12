PYTHON = python3.9
PIP = pip3.9
GIT = git 

TWEEPY_HASH = 1a5ba74
# .PHONY defines parts of the makefile that are not dependant on any specific file
# This is most often used to store functions
.PHONY = help requirements-conda dev-requirements-conda test test-twitter test-reddit test-graph test-score test-tweepy test-model run # clean

# Defines the default target that `make` will to try to make, or in the case of a phony target, execute the specified commands
# This target is executed whenever we just type `make`
.DEFAULT_GOAL = help

# The @ makes sure that the command itself isn't echoed in the terminal
help:
	@echo "---------------HELP-----------------"
	@echo "To install dependencies type make requirements-conda"
	@echo "To install dev dependencies type make dev-requirements-conda"
	@echo "To test the project type make test"
	@echo "To run the project type make run"
	@echo "------------------------------------"

requirements-conda:
	conda install -y -c conda-forge --file requirements.txt
	
	# install custom tweepy version
	${GIT} clone https://github.com/tweepy/tweepy || true
	cd tweepy && ${GIT} checkout 1a5ba74 && ${PYTHON} setup.py install
	
	# wptools is not available in the conda channels
	pip install wptools

dev-requirements-conda: requirements-conda
	conda install -y -c conda-forge pytest

test:
	${PYTHON} -m pytest

test-twitter:
	${PYTHON} -m pytest tests/test_twitter_collector.py

test-reddit:
	${PYTHON} -m pytest tests/test_reddit_collector.py

test-graph:
	${PYTHON} -m pytest tests/test_graph.py

test-tweepy:
	${PYTHON} -m pytest tests/test_tweepy.py

test-score:
	${PYTHON} -m pytest tests/test_score.py

test-model:
	PYTHONPATH=`pwd`
	${PYTHON} tests/test_synthetic.py

run:
	${PYTHON} main.py

# In this context, the *.project pattern means "anything that has the .project extension"
# clean:
#     rm -r *.project
