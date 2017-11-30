# BiLSTM_MMLSPARK
I want to integrate CNTK with spark using mmlspark. For this project i have used a RNN network, precisely a Bilateral LSTM network. I refer this notebook:

https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_202_Language_Understanding.ipynb

where i have taken the code and the model. Then I have integrate mmlspark function to get same output than normal CNTK evaluation.

Run this command to install and to make this software work:

spark-submit --packages Azure:mmlspark:0.10 cntk_mmlspark.py

CNTK documentation:

https://cntk.ai/pythondocs/gettingstarted.html

MMLSPARK documentation:

https://github.com/Azure/mmlspark

