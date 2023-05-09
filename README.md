# Text generation
The Python script included in this repository contains a couple of for the creation of toy text generation models. The first one is an n-gram model implemented from scratch, and the seccond a Long Short-Term Memory (LSTM) network-based one.
A Jupyter Notebook is included describing the creation and/or training process of the models, as well as the generation of some texts based on a couple of corpus; which are also included. 

## Key features
* Process a corpus
* Create n-gram and LSTM-based models for text generation
* Control parameters such as:
  *  Maximum number of tokens to be generated before halting 
  *  Cutoff for the probability distibution to only consider the top tokens, as a creativity control 
  *  Maximum vovabulary size to control complexity
* Train the the LSTM-based model on the processed corpus
* Model testing using perplexity as a metric
* Text generation
* Saving and loading models

## Some examples of generated text 
* `i love the commonwealth - canada , 1 1 march 2 0 0 2 2 is not a year of the diversity that has necessarily kept people apart has , quite simply` (stopped when 30 tokens were generated)
* `i love the commonwealth ' s leaders , as evident in australia last week ; and to share in the ideals of this unique gathering of nations , to celebrate an` (stopped when 30 tokens were generated)
* `where are you father flynn ?” said mr o ’ connor .`
* `the english lady waited in the window .`
* `ireland is german .`
