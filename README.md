# CNTK with MMLSPARK
I want to integrate spark in CNTK using mmlspark. In this project I have used a RNN network, precisely a Bilateral LSTM network and I refer this notebook to fetch the model and code:

https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_202_Language_Understanding.ipynb

The task of this notebook is slot tagging (tag each individual words to their respective classes) for the Air Travel Information Services (ATIS). 

Integrating mmlspark functions I can get same output than normal evaluation with CNTK, but I had to process the dataset to ensure that every word recognizes the context of the sentence. This because the notebook uses a (lightly preprocessed) version of the ATIS dataset. The network's input becames a window of five words that corrisponding with two preceding words and two following words of the word that I want process. 
Each word of this window I trasform it to one-hot array with words's vocabolary size storing all arrays like a matrix and, after all, I feed the model with the matrix flatten like input. The label is a one-hot array with labels dictionary size.

An example:

sentence = BOS i need a flight on monday from columbus to minneapolis at 838 am EOS

result with mmlspark:  [('BOS', 'O'), ('i', 'O'), ('need', 'O'), ('a', 'O'), ('flight', 'O'), ('on', 'O'), ('monday', 'B-depart_date.day_name'), ('from', 'O'), ('columbus', 'B-fromloc.city_name'), ('to', 'O'), ('minneapolis', 'B-toloc.city_name'), ('at', 'O'), ('838', 'B-depart_time.time'), ('am', 'I-depart_time.time'), ('EOS', 'O')]

To run this code with mmlspark:

spark-submit --packages Azure:mmlspark:0.10 cntk_mmlspark_BiLSTM.py

or with normal CNTK:

python cntk_BiLSTM.py


CNTK documentation:

https://cntk.ai/pythondocs/gettingstarted.html

MMLSPARK documentation:

https://github.com/Azure/mmlspark

