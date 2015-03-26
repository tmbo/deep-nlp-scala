# deep-nlp-scala
Using deep learning to POS tag sentences using scala + DL4J.

This is a showcase repository intended to evaluate different algorithms on the task of POS tagging german sentences. There is a Multilayerperceptron (from scratch), a Hidden-Markov-Model (from scratch) and a RBF deep net (based on DL4J) implementation.

## Installation
To execute the project one needs to make sure sbt 0.13 and java >= 1.6 is installed.

Information about how to install sbt on your system can be found on http://www.scala-sbt.org/release/tutorial/Setup.html

## Assets
Assets need to be placed into the assets/ directory. 

There should be labeled training data: de-train.tt, de-test.tt and de-eval.tt

And there should be unlabeled training data for the word2vec training in the folder assets/deu_news_2010_1M-text. The
data set can be downloaded from http://corpora.uni-leipzig.de/download.html


## Running
Running the different methods requires passing either 'mlp', 'rbm', 'hmm' or 'hmm-s' to the executable. Make sure to allow the JVM to use as much memory as possible (e.g. using "-Xmx7G").

The program can be started using "sbt run mlp"
