
import build_dataset as B
import training_evaluating_model as TE

#===================================================================================
#================================= Run Train and Evaluation ========================

# number of epoch
max_epochs=11

# load the model for epoch
model_path = "model_{}.cntk".format(max_epochs-1)
xx=TE.Modelclass(vocabolary_size=943*5, n_label=129, embed_dim=943, hid_dim=300, n_epochs=max_epochs, sparse=False)
if TE.os.path.isfile(model_path):
    z = TE.C.Function.load(model_path)
    print("Exist the model  "+model_path)
else:
    xx.do_train(B.my_dataset_train)
    xx.do_test(B.my_dataset_test)
    z = TE.C.Function.load(model_path)
print("parameters= ",z.arguments)
print("output= ",z.outputs)
#print(z.classify.b.value)

#====================================================================================
#================ IMPORT MMLSPARK AND SPARK LIBRARY =================================

from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType, ArrayType, FloatType, StringType, DoubleType
from pyspark.sql import Row
from pyspark.sql.session import SparkSession
from pyspark import SparkConf
from pyspark.context import SparkContext
from mmlspark import CNTKModel

conf= SparkConf().setMaster("local").setAppName("SequenceToSequence")
sc = SparkContext(conf = conf)
spark = SparkSession(sc)

# let's run a sequence through
#seq="BOS i want to fly from san francisco at 838 am and arrive in denver at 1110 in the morning EOS"
#seq = 'BOS i need a flight tomorrow from columbus to minneapolis at 838 am EOS'
seq='BOS show flights from burbank to st. louis on monday EOS'
print("sentence= ", seq)
length=len(seq.split())

#================================ RUN WITH MMLSPARK ==================================
#=====================================================================================

# fetch word from sentence and create the array flatten of the windows
token=[]
window=[]
for idx,words in enumerate(seq.split()):
  window.append(B.query_dict[words])

for i,int_word in enumerate(window):
  length=len(window) 
  token.append(B.create_batch5(i, length, window, int_word, 943).tolist()) 
       

df = spark.createDataFrame(enumerate(token), ["index","batchs"])

# cast element from double to float
def to_float(item):
  tmp = []
  for i in item:
    tmp.append(float(i))
  return tmp  

one_hot = udf(to_float, ArrayType(FloatType()))
df = df.withColumn("features", one_hot("batchs"))#.select("features")
df.printSchema()

cntkModel = CNTKModel().setModelLocation(spark, model_path )\
                   .setOutputNodeName("classify") \
                   .setInputCol("features") \
                   .setOutputCol("prob") \
                   .setMiniBatchSize(1)

df = cntkModel.transform(df).cache()
df.show()
def probsToEntities(probs):
    nn = B.slots_wl[B.np.argmax(probs)] 
    return nn

toEntityUDF = udf(probsToEntities,StringType())
df = df.withColumn("entities", toEntityUDF("prob"))

ent=[i.entities for i in df.select('entities').collect()]
result = list(zip(seq.split(),ent))
print(" ")
print("result with mmlspark: ", result)
print(" ")



