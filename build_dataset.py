from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import requests
import os
import numpy as np

#====================================================================================
#========================== DOWNLOAD AND BUILD DATASET ==============================

def download(url, filename):
    """ utility function to download a file """
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        for data in response.iter_content():
            handle.write(data)
locations = ['Tutorials/SLUHandsOn', 'Examples/LanguageUnderstanding/ATIS/BrainScript']
my_dataset_train = 'dataset_train_rewrited.ctf'
my_dataset_test = 'dataset_test_rewrited.ctf'

data = {
  'train': { 'file': 'atis.train.ctf', 'location': 0 },
  'test': { 'file': 'atis.test.ctf', 'location': 0 },
  'query': { 'file': 'query.wl', 'location': 1 },
  'slots': { 'file': 'slots.wl', 'location': 1 },
  'intent': { 'file': 'intent.wl', 'location': 1 }  
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

# load dictionaries
query_wl = [line.rstrip('\n') for line in open(data['query']['file'])]
slots_wl = [line.rstrip('\n') for line in open(data['slots']['file'])]
query_dict = {query_wl[i]:i for i in range(len(query_wl))}
slots_dict = {slots_wl[i]:i for i in range(len(slots_wl))}        

# it is possible to build the new dataset with this functions. To ensure that every word recognizes the context 
# of sentence, I consider the two preceding words and the two following words of my token in the sentence. For 
# this aim I use a window of 5 words, for each word of this window I create one-hot vector with words dictionary 
# size. After all I feed the model with the matrix of one-hot arrays flatten of this window like input. The label 
# is a one-hot array with labels dictionary size.

def create_one_hot_vector( num,length ):
    one_hot = np.zeros(length, np.float32)
    one_hot[int(num)] = 1
    return one_hot

def create_batch5(i, length_s, window, word, length_one_hot):
    Xf = np.zeros((5, length_one_hot))
    if i==0:   
        Xf[2] = create_one_hot_vector(word, length_one_hot)
        Xf[3] = create_one_hot_vector(window[i+1], length_one_hot)
        if length_s > 2:
          Xf[4] = create_one_hot_vector(window[i+2], length_one_hot)
    elif i==1:
        Xf[1] = create_one_hot_vector(window[i-1], length_one_hot)
        Xf[2] = create_one_hot_vector(word, length_one_hot)
        if length_s > 2:
          Xf[3] = create_one_hot_vector(window[i+1], length_one_hot)
        if length_s > 3:
          Xf[4] = create_one_hot_vector(window[i+2], length_one_hot)
    elif i==length_s-2:
        if length_s > 3:
          Xf[0] = create_one_hot_vector(window[i-2], length_one_hot)
        if length_s > 2:
          Xf[1] = create_one_hot_vector(window[i-1], length_one_hot)
        Xf[2] = create_one_hot_vector(word, length_one_hot)
        Xf[3] = create_one_hot_vector(window[i+1], length_one_hot)
    elif i==length_s-1:
        if length_s > 2:  
          Xf[0] = create_one_hot_vector(window[i-2], length_one_hot)
        Xf[1] = create_one_hot_vector(window[i-1], length_one_hot)
        Xf[2] = create_one_hot_vector(word, length_one_hot)
    else:
        Xf[0] = create_one_hot_vector(window[i-2], length_one_hot)
        Xf[1] = create_one_hot_vector(window[i-1], length_one_hot)
        Xf[2] = create_one_hot_vector(word, length_one_hot)
        Xf[3] = create_one_hot_vector(window[i+1], length_one_hot)
        Xf[4] = create_one_hot_vector(window[i+2], length_one_hot)
    return Xf.flatten()

def writefile(namefile, namedataset):  
# here I fetch the index of the words inside the sentences of the input of the dataset
    features = []
    temp = []
    index=0
    with open(namedataset,'r') as f: 
      for line in f:
        items = line.split()
        if(items[0]==index): 
          temp.append(items[2].split(':')[0])
        else:
          index = items[0]
          features.append(temp)
          temp= []
          temp.append(items[2].split(':')[0])
      features.append(temp)
    features = features[1:]
# here I fetch the index of the labels inside the sentences of the dataset
    labels = []
    temp1 = []
    index1=0
    with open(namedataset,'r') as f: 
      for line in f:
        items = line.split()
        if(items[0]==index1):
          temp1.append(items[6].split(':')[0] if items[5]!='|S1' else items[10].split(':')[0])
        else:
          index1 = items[0]
          labels.append(temp1)
          temp1= []
          temp1.append(items[6].split(':')[0] if items[5]!='|S1' else items[10].split(':')[0])
      labels.append(temp1)
    labels = labels[1:]
  
# open file in write mode and I write each row with one-hot matrix flatten and label
    out_file = open(namefile,"w")
    for num,item in enumerate(features):
      size=len(item)
      for i,word in enumerate(item):
        file_feature=create_batch5(i,size,item,word, len(query_wl))
        out_file.write("|S0 ")
        for j in file_feature:
          out_file.write(str(j)+" ")
        out_file.write("|S2 ")
        file_label=create_one_hot_vector(labels[num][i], len(slots_wl))
        for j in file_label:
          out_file.write(str(j)+" ")
        out_file.write("\n")    
    out_file.close()


if os.path.exists(my_dataset_train) and os.path.exists(my_dataset_test):
  print("File dataset exist: {}".format(my_dataset_train))
  print("File dataset exist: {}".format(my_dataset_test))
else:
  writefile(my_dataset_train, data['train']['file'])
  writefile(my_dataset_test, data['test']['file'])