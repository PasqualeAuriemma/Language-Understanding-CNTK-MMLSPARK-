#================= IMPORTING FILE =================================================

import training_evaluating_model as TE
import build_dataset as B

#===================================================================================
#================================= Run Train and Evaluation ========================

max_epochs=10

# load the model for epoch
model_path = "model_{}.cntk".format(max_epochs-1)
xx=TE.Modelclass(vocabolary_size=943, n_label=129, embed_dim=150, hid_dim=300, n_epochs=max_epochs, sparse=True)

if TE.os.path.isfile(model_path):
    z = TE.C.Function.load(model_path)
    print("Exist the model  "+model_path)
else:    
    xx.do_train(B.data['train']['file'])
    xx.do_test(B.data['test']['file'])
    z = TE.C.Function.load(model_path)
print("parameters= ",z.arguments)
print("output= ",z.outputs)
#print(z.classify.b.value)

# let's run a sequence through
#seq="BOS i want to fly from san francisco at 838 am and arrive in denver at 1110 in the morning EOS"
seq = 'BOS show flights from burbank to st. louis on monday EOS'
print("sentence= ", seq)

#====================================================================================
#========================== TEST MODEL WITH CNTK NORMAL =============================

w = [B.query_dict[w] for w in seq.split()] # convert to word indices
print("w= ",w)

onehot = B.np.zeros([len(w),len(B.query_dict)], B.np.float32)
for t in range(len(w)):
    onehot[t,w[t]] = 1

# remember input: x = C.sequence.input_variable(vocab_size, dtype=np.float32)
# onehot can be python list or one hot numpy
pred = z(xx.x).eval({xx.x:[onehot]})[0]
best = B.np.argmax(pred,axis=1)
print(" ")
print("result without mmlspark",list(zip(seq.split(),[B.slots_wl[s] for s in best])))

#====================================================================================
