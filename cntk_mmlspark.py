from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import requests
import os
import numpy as np
#================ IMPORT MMLSPARK AND SPARK LIBRARY =================================

from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType, ArrayType, FloatType, StringType, DoubleType
from pyspark.sql import Row
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql.session import SparkSession
from pyspark import SparkConf
from pyspark.context import SparkContext
from mmlspark import CNTKModel, ModelDownloader, AssembleFeatures

conf= SparkConf().setMaster("local").setAppName("SequenceToSequence")
sc = SparkContext(conf = conf)
spark = SparkSession(sc)
#====================================================================================
#========================== DOWNLOAD DATASET ========================================

def download(url, filename):
    """ utility function to download a file """
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        for data in response.iter_content():
            handle.write(data)
locations = ['Tutorials/SLUHandsOn', 'Examples/LanguageUnderstanding/ATIS/BrainScript']
#train_exa.txt    atis.train.ctf example_test.ctf exam_test.ctf
data = {
  'train': { 'file': 'atis.train.ctf', 'location': 0 },
  'test': { 'file': 'atis.test.ctf', 'location': 0 },
  'query': { 'file': 'query.wl', 'location': 1 },
  'slots': { 'file': 'slots.wl', 'location': 1 },
  'intent': { 'file': 'intent.wl', 'location': 1 }  
} 

data1 = {
  'train': { 'file': 'last_test5.ctf', 'location': 0 },
  'train2': { 'file': 'last_test1.ctf', 'location': 0 },
  'train3': { 'file': 'last_test_prova.ctf', 'location': 0 },
  'train4': { 'file': 'last_test_final.ctf', 'location': 0 },
  'train1': { 'file': 'last_test3.ctf', 'location': 0 }
}

for item in data.values():
    location = locations[item['location']]
    path = os.path.join('..', location, item['file'])
    if os.path.exists(path):
        print("Reusing locally cached:", item['file'])
        # Update path
        item['file'] = path
    elif os.path.exists(item['file']):
        print("Reusing locally cached:", item['file'])
    else:
        print("Starting download:", item['file'])
        url = "https://github.com/Microsoft/CNTK/blob/release/2.2/%s/%s?raw=true"%(location, item['file'])
        download(url, item['file'])
        print("Download completed")


# I try the correct dataset

def one_hot_vector( num,length ):
    one_hot = np.zeros(length, np.float32)
    one_hot[int(num)] = 1
    return one_hot

def create_batch(i,length,item,word, num):
    Xf = np.zeros((1, num))
    if i==0:   
        Xf[0] = one_hot_vector(word,num)
    elif i==1:
        Xf[0] = one_hot_vector(word,num)#2
    elif i==length-2:
        Xf[0] = one_hot_vector(word,num)#7
    elif i==length-1:
        Xf[0] = one_hot_vector(word,num)#11
    else:
        Xf[0] = one_hot_vector(word,num)#14
    return Xf.flatten()

def create_batch5(i,length,item,word, num):
    Xf = np.zeros((5, num))
    if i==0:   
        Xf[2] = one_hot_vector(word,num)
        Xf[3] = one_hot_vector(item[i+1],num)
        if length > 2:
          Xf[4] = one_hot_vector(item[i+2],num)
    elif i==1:
        Xf[1] = one_hot_vector(item[i-1],num)#1
        Xf[2] = one_hot_vector(word,num)#2
        if length > 2:
          Xf[3] = one_hot_vector(item[i+1],num)#3
        if length > 3:
          Xf[4] = one_hot_vector(item[i+2],num)#4
    elif i==length-2:
        if length > 3:
          Xf[0] = one_hot_vector(item[i-2],num)#5
        if length > 2:
          Xf[1] = one_hot_vector(item[i-1],num)#6
        Xf[2] = one_hot_vector(word,num)#7
        Xf[3] = one_hot_vector(item[i+1],num)#8
    elif i==length-1:
        if length > 2:  
          Xf[0] = one_hot_vector(item[i-2],num)#9
        Xf[1] = one_hot_vector(item[i-1],num)#10
        Xf[2] = one_hot_vector(word,num)#11
    else:
        Xf[0] = one_hot_vector(item[i-2],num)#12
        Xf[1] = one_hot_vector(item[i-1],num)#13
        Xf[2] = one_hot_vector(word,num)#14
        Xf[3] = one_hot_vector(item[i+1],num)#15
        Xf[4] = one_hot_vector(item[i+2],num)#16  
    return Xf.flatten()
if os.path.exists(data1['train4']['file']):
  print("existent file")
else:
  #features
  featur = []
  temp = []
  index=0
  with open(data['train']['file'],'r') as f: 
    for line in f:
      items = line.split()
      if(items[0]==index): 
        temp.append(items[2].split(':')[0])
      else:
        index = items[0]
        featur.append(temp)
        temp= []
        temp.append(items[2].split(':')[0])
    featur.append(temp)
  featur = featur[1:]
  # label
  labe = []
  temp1 = []
  index1=0
  with open(data['train']['file'],'r') as f: 
    for line in f:
      items = line.split()
      if(items[0]==index1):
        temp1.append(items[6].split(':')[0] if items[5]!='|S1' else items[10].split(':')[0])
      else:
        index1 = items[0]
        labe.append(temp1)
        temp1= []
        temp1.append(items[6].split(':')[0] if items[5]!='|S1' else items[10].split(':')[0])
    labe.append(temp1)
  labe = labe[1:]
  #------------------------------- Write --------------
  out_file = open(data1['train4']['file'],"w")
  #print(len(featur), " ",len(featur[0]))
  for num,item in enumerate(featur):
    size=len(item)
    print(num, " ", size)
    for i,word in enumerate(item):
      file_feature=create_batch5(i,size,item,word,943)
      out_file.write("|S0 ")
      for j in file_feature:
        out_file.write(str(j)+" ")
    #for j,word in enumerate(labe[num]):
      out_file.write("|S2 ")
      file_label=one_hot_vector(labe[num][i],129)
      for j in file_label:
        out_file.write(str(j)+" ")
      out_file.write("\n")
    break    
  out_file.close()


#==============================================================================
#================= IMPORTING LIBRARIES ========================================

import math
import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

#===============================================================================
#============================= NETWORK DESCRIPTION =============================

# number of words in vocab, slot labels, and intent labels
vocab_size = 943*5 ; num_labels = 129;    

# model dimensions
input_dim  = vocab_size
label_dim  = num_labels
emb_dim    = 943
hidden_dim = 300
max_epochs = 70


# Create the containers for input feature (x) and the label (y)
x = C.sequence.input_variable(vocab_size, dtype=np.float32, name="in")
y = C.sequence.input_variable(num_labels, dtype=np.float32, name="y")

#================= MODELL BIDIRECTIONAL ========================================

def BiRecurrence(fwd, bwd):
    F = C.layers.Recurrence(fwd)
    G = C.layers.Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = C.splice(F(x), G(x))
    return apply_x 

def create_model():
    with C.layers.default_options(initial_state=0.1):
        model= C.layers.Sequential([
            C.layers.Embedding(emb_dim, name='embed'),
            BiRecurrence(C.layers.LSTM(hidden_dim//2), 
                                  C.layers.LSTM(hidden_dim//2)),
            C.layers.Dense(num_labels, name='classify')
        ])
        return model


#==============================================================================
#========================= CREATE READER FROM CNTKTextFormatReader ============

def create_reader(path, is_training):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
         query         = C.io.StreamDef(field='S0', shape=vocab_size,  is_sparse=False),           
         slot_labels   = C.io.StreamDef(field='S2', shape=num_labels,  is_sparse=False)
     )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)
#intent        = C.io.StreamDef(field='S1', shape=num_intents, is_sparse=True),
# peek
reader = create_reader(data1['train3']['file'], is_training=True)
reader.streams.keys()

#================================================================================
#===================== CREATE CRITERION FUNCTION ================================



def create_criterion_function(model, labels):
    ce   = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error      (model, labels)
    return ce, errs # (model, labels) -> (loss, error metric)


#=================================================================================
#======================== TRAINING ===============================================

def train(reader, model_func, max_epochs=max_epochs):
    
    # Instantiate the model function; x is the input (feature) variable 
    model = model_func(x)
    
    # Instantiate the loss and error function
    loss, label_error = create_criterion_function(model, y)

    # training config
    epoch_size = 36000       # 18000 samples is half the dataset size 
    minibatch_size = 72
    
    # LR schedule over epochs 
    # In CNTK, an epoch is how often we get out of the minibatch loop to
    # do other stuff (e.g. checkpointing, adjust learning rate, etc.)
    lr_per_sample = [3e-4]*4+[1.5e-4]
    lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
    lr_schedule = C.learning_rate_schedule(lr_per_minibatch, C.UnitType.minibatch, epoch_size)
    
    # Momentum schedule
    momentum_as_time_constant = C.momentum_as_time_constant_schedule(700)
    
    # We use a the Adam optimizer which is known to work well on this dataset
    # Feel free to try other optimizers from 
    # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner
    learner = C.adam(parameters=model.parameters,
                     lr=lr_schedule,
                     momentum=momentum_as_time_constant,
                     gradient_clipping_threshold_per_sample=15, 
                     gradient_clipping_with_truncation=True)

    # Setup the progress updater
    progress_printer = C.logging.ProgressPrinter(tag='Training',num_epochs=max_epochs)
    
    # Uncomment below for more detailed logging
    #progress_printer = ProgressPrinter(freq=100, first=10, tag='Training', num_epochs=max_epochs) 

    # Instantiate the trainer
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)

    # process minibatches and perform model training
    C.logging.log_number_of_parameters(model)
    
    # Assign the data fields to be read from the input
    
    data_map={x: reader.streams.query, y: reader.streams.slot_labels}
    
    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        print("epoch= ",epoch)
        p=0
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = reader.next_minibatch(minibatch_size, input_map= data_map)  # fetch minibatch
            #print("data= ", data)
            trainer.train_minibatch(data)               # update model with it
            t += data[y].num_samples                    # samples so far
            print(epoch,"p= ",t+p)
        trainer.summarize_training_progress()
        model_path = "model_%d.cntk" % epoch
    print("Saving final model to '%s'" % model_path)
    model.save(model_path)
    print("%d epochs complete." % max_epochs)



#===================================================================================        
#===================================== Run the trainer =============================

def do_train():
    global z
    z = create_model()
    reader = create_reader(data1['train4']['file'], is_training=True)
    train(reader, z)


# load the model for epoch 0
model_path = "model_{}.cntk".format(max_epochs-1)

if os.path.isfile(model_path):
    z = C.Function.load(model_path)
    print("Exist the model  "+model_path)
else:
    do_train()

print("parameters= ",z.arguments)
print("output= ",z.outputs)


#===================================================================================
#================================== EVALUATION =====================================

def evaluate(reader, model_func):
    
    # Instantiate the model function; x is the input (feature) variable 
    model = model_func(x)
    
    # Create the loss and error functions
    loss, label_error = create_criterion_function(model, y)

    # process minibatches and perform evaluation
    progress_printer = C.logging.ProgressPrinter(tag='Evaluation', num_epochs=0)
    
    # Assign the data fields to be read from the input
    data_map={x: reader.streams.query, y: reader.streams.slot_labels}
    
    while True:
        minibatch_size = 500
        data = reader.next_minibatch(minibatch_size, input_map= data_map)  # fetch minibatch
        if not data:                                 # until we hit the end
            break

        evaluator = C.eval.Evaluator(loss, progress_printer)
        evaluator.test_minibatch(data)
 
    evaluator.summarize_test_progress()

#===================================================================================
#================================= Run Evaluation ==================================

def do_test():
    reader = create_reader(data['test']['file'], is_training=False)
    evaluate(reader, z)
#do_test()
#print(z.classify.b.value)

#====================================================================================

# load dictionaries
query_wl = [line.rstrip('\n') for line in open(data['query']['file'])]
slots_wl = [line.rstrip('\n') for line in open(data['slots']['file'])]
query_dict = {query_wl[i]:i for i in range(len(query_wl))}
slots_dict = {slots_wl[i]:i for i in range(len(slots_wl))}

# let's run a sequence through

#seq="BOS i want to fly from san francisco at 838 am and arrive in denver at 1110 in the morning EOS"
seq = 'BOS zone from seattle at 838 am to san francisco monday EOS'
print("sentence= ", seq)


#================================ RUN WITH MMLSPARK ==================================
#=====================================================================================

token=[]
window=[]
for idx,words in enumerate(seq.split()):
  window.append(query_dict[words])

for i,int_word in enumerate(window):
  length=len(window) 
  token.append(create_batch5(i, length, window, int_word, 943).tolist()) 
       

df = spark.createDataFrame(enumerate(token), ["index","batchs"])


def to_float(item):
  tmp = []
  for i in item:
    tmp.append(float(i))
  return tmp  

one_hot = udf(to_float, ArrayType(FloatType()))
df = df.withColumn("features", one_hot("batchs")).select("features")

df.printSchema()

location = "/home/sod/Scrivania/"+ model_path

cntkModel = CNTKModel().setModelLocation(spark, location )\
                   .setOutputNodeName("classify") \
                   .setInputCol("features") \
                   .setOutputCol("prob") \
                   .setMiniBatchSize(1)

df = cntkModel.transform(df).cache()

def probsToEntities(probs):
    reshaped_probs = np.array(probs).reshape(1, 129)
    nn = [slots_wl[np.argmax(probs)] for probs in reshaped_probs]
    return nn[0]

toEntityUDF = udf(probsToEntities,StringType())
df = df.withColumn("entities", toEntityUDF("prob"))

ent=[i.entities for i in df.select('entities').collect()]
result = list(zip(seq.split(),ent))
print(" ")
print("result with mmlspark: ", result)
print(" ")

#========================== RUN WITHOUT MMLSPARK ====================================
#====================================================================================

w = [query_dict[w] for w in seq.split()] # convert to word indices
#print("w= ",w)

# remember input: x = C.sequence.input_variable(vocab_size, dtype=np.float32)
# onehot can be python list or one hot numpy
pred = z(x).eval({x:[token]})[0]

best=[]
for pp in pred:
  reshaped_probs = np.array(pp).reshape(1, 129)
  nn = [np.argmax(probs) for probs in reshaped_probs]
  best.append(nn[0])

print(" ")
print("result without mmlspark: ",list(zip(seq.split(),[slots_wl[s] for s in best])))

#====================================================================================




