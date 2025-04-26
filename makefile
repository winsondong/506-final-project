# Makefile for Airbnb price-prediction 

.PHONY: install run visualize all clean

# install your dependencies
install:
	pip install -r requirements.txt

# run your combined RF + XGB script
run:
	python Final_project_RandomForest.py

# regenerate your plots
visualize:
	python visualizing.py

# do everything in one go
all: install run visualize

# clean up Python byte-code
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
