# sentiment_analysis

Clone the repo to your local machine to ensure files are in the correct location.

The analysis used Keras and Tensorflow, which currently only supports Python 3.6 or lower. I trained the RNNs on my GPU, if you would like to do the same, I recommend following these instructions https://medium.com/@ab9.bhatia/set-up-gpu-accelerated-tensorflow-keras-on-windows-10-with-anaconda-e71bfa9506d1

## File List

Sentiment Classifer.pdf is an overview of the entire project and a good starting point.

Sentiment Analysis.ipynb is a jupyter notebook that performs the full data science pipeline from intake to modeling.

sentiment_preprocessing.py - prepares the data for modeling. Must be run before running sentiment_ML_models.py

sentiment_ML_models.py - creates traditional ML models and RNN models and saves them to disk alongside their results.

EDAhelperfunctions.py - a module of helper functions for EDA.

requirements.txt is a list of dependencies and versions.
