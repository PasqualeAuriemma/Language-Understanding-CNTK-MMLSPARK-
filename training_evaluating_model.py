#================= IMPORTING FILE =============================================

import build_dataset as B

#==============================================================================
#================= IMPORTING LIBRARIES ========================================

import numpy as np
import os
import math
import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

#===============================================================================
#============================= NETWORK DESCRIPTION =============================

class Modelclass:

# number of words in vocab, slot labels, and intent labels    
#  vocab_size = 943
#  num_labels = 129    

# model dimensions
#  emb_dim    = 150
#  hidden_dim = 300
#  max_epochs = 0

  def __init__(self, vocabolary_size, n_label, embed_dim, hid_dim, n_epochs, sparse=False):
    self.vocab_size = vocabolary_size #943*5 
    self.num_labels = n_label #129    
    self.emb_dim    = embed_dim #943
    self.hidden_dim = hid_dim #300
    self.max_epochs = n_epochs #1
    self.sparse=sparse
    self.x = C.sequence.input_variable(self.vocab_size, dtype=np.float32, name="in")
    self.y = C.sequence.input_variable(self.num_labels, dtype=np.float32, name="y")

  

#===============================================================================
#================= MODELL BIDIRECTIONAL ========================================

  def BiRecurrence(self, fwd, bwd):
    F = C.layers.Recurrence(fwd)
    G = C.layers.Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = C.splice(F(x), G(x))
    return apply_x 

  def create_model(self):
    with C.layers.default_options(initial_state=0.1):
        model= C.layers.Sequential([
            C.layers.Embedding(self.emb_dim, name='embed'),
            self.BiRecurrence(C.layers.LSTM(self.hidden_dim//2), 
                                  C.layers.LSTM(self.hidden_dim//2)),
            C.layers.Dense(self.num_labels, name='classify')
        ])
        return model

#==============================================================================
#========================= CREATE READER FROM CNTKTextFormatReader ============

  def create_reader(self, path, is_training, sparse):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
         query         = C.io.StreamDef(field='S0', shape=self.vocab_size,  is_sparse=sparse),           
         slot_labels   = C.io.StreamDef(field='S2', shape=self.num_labels,  is_sparse=sparse)
     )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

#================================================================================
#===================== CREATE CRITERION FUNCTION ================================

  def create_criterion_function(self, model, labels):
    ce   = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error      (model, labels)
    return ce, errs # (model, labels) -> (loss, error metric)

#=================================================================================
#======================== TRAINING ===============================================

  def train(self, reader, model_func):
    
    # Instantiate the model function; x is the input (feature) variable 
    model = model_func(self.x)
    
    # Instantiate the loss and error function
    loss, label_error = self.create_criterion_function(model, self.y)

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
    # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner (momentum_sgd foe example)
    learner = C.adam(parameters=model.parameters,
                     lr=lr_schedule,
                     momentum=momentum_as_time_constant,
                     gradient_clipping_threshold_per_sample=15, 
                     gradient_clipping_with_truncation=True)

    # Setup the progress updater
    progress_printer = C.logging.ProgressPrinter(tag='Training',num_epochs=self.max_epochs)
    
    # Uncomment below for more detailed logging
    #progress_printer = ProgressPrinter(freq=100, first=10, tag='Training', num_epochs=max_epochs) 

    # Instantiate the trainer
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)

    # process minibatches and perform model training
    C.logging.log_number_of_parameters(model)
    
    # Assign the data fields to be read from the input
    
    data_map={self.x: reader.streams.query, self.y: reader.streams.slot_labels}
    
    t = 0
    for epoch in range(self.max_epochs):         # loop over epochs
        p=0
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = reader.next_minibatch(minibatch_size, input_map= data_map)  # fetch minibatch
            trainer.train_minibatch(data)               # update model with it
            t += data[self.y].num_samples                    # samples so far
            #print(epoch,"p= ",t+p)
        trainer.summarize_training_progress()
        model_path = "model_%d.cntk" % epoch
    print("Saving final model to '%s'" % model_path)
    model.save(model_path)
    print("%d epochs complete." % self.max_epochs)

#===================================================================================
#================================== EVALUATION =====================================

  def evaluate(self, reader, model_func):
    
    # Instantiate the model function; x is the input (feature) variable 
    model = model_func(self.x)
    
    # Create the loss and error functions
    loss, label_error = self.create_criterion_function(model, self.y)

    # process minibatches and perform evaluation
    progress_printer = C.logging.ProgressPrinter(tag='Evaluation', num_epochs=0)
    
    # Assign the data fields to be read from the input
    data_map={self.x: reader.streams.query, self.y: reader.streams.slot_labels}
    
    while True:
        minibatch_size = 500
    # fetch minibatch    
        data = reader.next_minibatch(minibatch_size, input_map= data_map) 
    # until we hit the end     
        if not data:                                 
            break

        evaluator = C.eval.Evaluator(loss, progress_printer)
        evaluator.test_minibatch(data)
 
    evaluator.summarize_test_progress()

#===================================================================================        
#====================== Function do trainer and Evaluation =========================

  def do_train(self, namefile):
    global z
    z = self.create_model()
    reader = self.create_reader(namefile, is_training=True, sparse=self.sparse)
    self.train(reader, z)

  def do_test(self, namefile):
    reader = self.create_reader(namefile, is_training=False, sparse=self.sparse)
    self.evaluate(reader, z)




