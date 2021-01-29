PYTHON = python3.9
PIP = pip3.9

# .PHONY defines parts of the makefile that are not dependant on any specific file
# This is most often used to store functions
.PHONY = help requirements dev-requirements test run # clean

# Defines the default target that `make` will to try to make, or in the case of a phony target, execute the specified commands
# This target is executed whenever we just type `make`
.DEFAULT_GOAL = help

# The @ makes sure that the command itself isn't echoed in the terminal
help:
	@echo "---------------HELP-----------------"
	@echo "To install dependencies type make requirements"
	@echo "To install dev dependencies type make dev-requirements"
	@echo "To test the project type make test"
	@echo "To run the project type make run"
	@echo "------------------------------------"

requirements:
	${PIP} install -r requirements

dev-requirements:
	${PIP} install -r requirements
	${PIP} install pytest

test:
	${PYTHON} -m pytest

run:
	${PYTHON} main.py

# In this context, the *.project pattern means "anything that has the .project extension"
# clean:
#     rm -r *.project