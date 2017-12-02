# BiLSTM_MMLSPARK
I want to integrate CNTK with spark using mmlspark. In this project I have used a RNN network, precisely a Bilateral LSTM network and I refer this notebook to fetch the model:

https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_202_Language_Understanding.ipynb

I have integrate mmlspark function in this notebook to get same output than normal CNTK evaluation.

To run this code with mmlspark:

spark-submit --packages Azure:mmlspark:0.10 cntk_mmlspark_BiLSTM.py

or with normal CNTK:

python cntk_BiLSTM.py


CNTK documentation:

https://cntk.ai/pythondocs/gettingstarted.html

MMLSPARK documentation:

https://github.com/Azure/mmlspark

