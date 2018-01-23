# CNTK with MMLSPARK
I want to integrate spark in CNTK using mmlspark. In this project I have used a RNN network, precisely a Bilateral LSTM network and I refer to this notebook to fetch the model and code:

https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_202_Language_Understanding.ipynb

The purpose of this notebook is slot tagging (tag each individual word to his class) and I have trained the model with two dataset: the Air Travel Information Services (ATIS) and sections of Wall Street Journal (WSJ). The first want to fetch some information from the sentence like departure or arrive station. The second want to tag all the words with their means in the sentence like name sentence, verb sentence, adjective or other.

Integrating mmlspark in CNTK I can get same outputs than normal evaluation with CNTK, but I had to modify the tokens, that the model take like input, to ensure that every word recognizes the context of the sentence. The network's input becames a window of five words that it contain two preceding words and two following words to the word that I want process. 
I consider each word of this window like to one-hot array of words's vocabolary dimension and storing their all arrays like a matrix and, after all, I feed the model with the matrix flatten like input. The label is a one-hot array of labels dictionary dimension.

An example of trained model with ATIS dataset:

sentence = BOS i wnat to know all flights from san francisco to kansas on mondey night at 1030 am EOS

result with mmlspark:  [('BOS', 'O'), ('i', 'O'), ('want', 'O'), ('to', 'O'), ('know', 'O'), ('all', 'O'), ('flights', 'O'), ('from', 'O'), ('san', 'B-fromloc.city_name'), ('francisco', 'I-fromloc.city_name'), ('to', 'O'), ('kansas', 'B-toloc.city_name'), ('city', 'I-toloc.city_name'), ('on', 'O'), ('monday', 'B-depart_date.day_name'), ('night', 'B-depart_time.period_of_day'), ('at', 'O'), ('1030', 'B-depart_time.time'), ('am', 'I-depart_time.time'), ('EOS', 'O')]

An example of trained model with WSJ dataset:

sentence = BOS next thursday the government will promise that the management of department will help all people to take confidence in the law EOS

result with mmlspark:  [('BOS', 'O'), ('next', 'b-np'), ('thursday', 'i-np'), ('the', 'b-np'), ('government', 'i-np'), ('will', 'b-vp'), ('promise', 'i-vp'), ('that', 'b-sbar'), ('the', 'b-np'), ('management', 'i-np'), ('of', 'b-pp'), ('department', 'b-np'), ('will', 'b-vp'), ('help', 'i-vp'), ('all', 'b-np'), ('people', 'i-np'), ('to', 'b-vp'), ('take', 'i-vp'), ('confidence', 'b-np'), ('in', 'b-pp'), ('the', 'b-np'), ('law', 'i-np'), ('EOS', 'O')]

To run this code with mmlspark:

spark-submit --packages Azure:mmlspark:0.10 cntk_mmlspark_BiLSTM.py

or with normal CNTK:

python cntk_BiLSTM.py


CNTK documentation:

https://cntk.ai/pythondocs/gettingstarted.html

MMLSPARK documentation:

https://github.com/Azure/mmlspark

