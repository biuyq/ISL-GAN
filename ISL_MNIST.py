# the code is highly based on the improved gans.
import sys
import argparse
import numpy as np
import theano as th
import theano.tensor as T
import lasagne
import lasagne.layers as LL
import time
import nn
from theano.sandbox.rng_mrg import MRG_RandomStreams

# settings
factor_M = 0.0
LAMBDA_2 = 0.5
prediction_decay = 0.5   
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--seed_data', type=int, default=2)
parser.add_argument('--unlabeled_weight', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--count', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.002)# learning rate, no decay
args = parser.parse_args()
print(args)


# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
data_rng = np.random.RandomState(args.seed_data)

# specify generative model
noise = theano_rng.uniform(size=(args.batch_size, 50))
gen_layers = [LL.InputLayer(shape=(args.batch_size, 50), input_var=noise)]
gen_layers.append(nn.batch_norm(LL.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=T.nnet.softplus), g=None))
gen_layers.append(nn.batch_norm(LL.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=T.nnet.softplus), g=None))
gen_layers.append(nn.l2normalize(LL.DenseLayer(gen_layers[-1], num_units=28**2, nonlinearity=T.nnet.sigmoid)))
gen_dat = LL.get_output(gen_layers[-1], deterministic=False)

# specify supervised model
layers = [LL.InputLayer(shape=(None, 28**2))]
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.3))
layers.append(nn.DenseLayer(layers[-1], num_units=1000))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=500))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=250))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=250))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=250))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=10, nonlinearity=None, train_scale=True))

# costs
labels = T.ivector()
x_lab = T.matrix()
x_unl = T.matrix()

training_targets =T.matrix('targets')
training_targets2 = T.matrix('targets2') 
training_targets3 = T.matrix('targets3') 


temp = LL.get_output(gen_layers[-1], deterministic=False, init=True)
temp = LL.get_output(layers[-1], x_lab, deterministic=False, init=True)
init_updates = [u for l in gen_layers+layers for u in getattr(l,'init_updates',[])]

output_before_softmax_lab = LL.get_output(layers[-1], x_lab, deterministic=False) # no softmax labeled dis output
output_before_softmax_unl,output_before_softmax_unl_ = LL.get_output([layers[-1],layers[-2]], x_unl, deterministic=False)  # last two layers' output 
output_before_softmax_gen = LL.get_output(layers[-1], gen_dat, deterministic=False) #dis of generator output 

l_lab = output_before_softmax_lab[T.arange(args.batch_size),labels] 
l_unl = nn.log_sum_exp(output_before_softmax_unl) 
l_unl_ = nn.log_sum_exp(output_before_softmax_unl_) 
l_gen = nn.log_sum_exp(output_before_softmax_gen)
#loss_lab = -T.mean(l_lab) + T.mean(T.mean(nn.log_sum_exp(output_before_softmax_lab)))
loss_lab = T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(output_before_softmax_lab),labels)*(T.exp((pow(T.nnet.softmax(output_before_softmax_lab)[T.arange(args.batch_size),labels],2.0)))))


#loss_ct = lasagne.objectives.squared_error(T.sum(T.exp(output_before_softmax_unl),axis =1)/(T.sum(T.exp(output_before_softmax_unl),axis =1)+1),T.sum(T.exp(output_before_softmax_unl2),axis=1)/(T.sum(T.exp(output_before_softmax_unl2),axis =1)+1))
loss_ct = T.mean(lasagne.objectives.squared_error(T.nnet.softmax(output_before_softmax_unl),T.nnet.softmax(training_targets)),axis = 1)
 
last_result = T.nnet.softmax(output_before_softmax_unl)

loss_ct_ = T.mean(lasagne.objectives.squared_error(output_before_softmax_unl_,training_targets2),axis = 1)    #D_ normalization, this term makes the model unstable

CT = LAMBDA_2*(loss_ct+0.0*loss_ct_)-factor_M
CT_ = T.mean(T.maximum(CT,0.0*CT),axis=0)

loss_unl = 0.5*(CT_ -T.mean(l_unl) + T.mean(T.nnet.softplus(l_unl)) -np.log(1) + T.mean(T.nnet.softplus(l_gen)))  #should be smaller


zeros = np.zeros(args.batch_size)
train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))
train_err2 = T.mean(T.le(T.max(output_before_softmax_lab,axis=1),zeros))  #mis-classification

m1 = T.mean(LL.get_output(layers[-3], gen_dat), axis=0)
m2 = T.mean(LL.get_output(layers[-3], x_unl), axis=0)
loss_gen = T.mean(T.square(m1 - m2))

# test error
output_before_softmax = LL.get_output(layers[-1], x_lab, deterministic=True)
test_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),labels))

# Theano functions for training and testing
lr = T.scalar()
disc_params = LL.get_all_params(layers, trainable=True)


disc_param_updates = nn.adam_updates(disc_params, loss_lab + args.unlabeled_weight*loss_unl, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
gen_params = LL.get_all_params(gen_layers, trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates)
train_batch_disc = th.function(inputs=[x_lab,labels,x_unl,training_targets,training_targets2,lr], outputs=[loss_lab, loss_unl, train_err,train_err2,output_before_softmax_unl,output_before_softmax_unl_,last_result], updates=disc_param_updates+disc_avg_updates)



train_batch_gen = th.function(inputs=[x_unl,lr], outputs=loss_gen, updates=gen_param_updates)
test_batch = th.function(inputs=[x_lab,labels], outputs=test_err, givens=disc_avg_givens)

# load MNIST data
data = np.load('mnist.npz')
trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(th.config.floatX)
trainx_unl = trainx.copy()
trainx_unl2 = trainx.copy()
trainy = np.concatenate([data['y_train'], data['y_valid']]).astype(np.int32)
nr_batches_train = int(trainx.shape[0]/args.batch_size)
testx = data['x_test'].astype(th.config.floatX)
testy = data['y_test'].astype(np.int32)
nr_batches_test = int(testx.shape[0]/args.batch_size)


trainx_unl_org = trainx.copy()
trainx_unl2_org = trainx.copy()

# select labeled data
inds = data_rng.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for j in range(10):
    txs.append(trainx[trainy==j][:args.count])
    tys.append(trainy[trainy==j][:args.count])
txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)

init_param(trainx[:500]) # data dependent initialization

# //////////// perform training //////////////
lr = 0.003                # learning rate 0.003

start_epoch = 0

training_targets = np.float32(np.zeros((len(trainx_unl_org), 10)))  # for saving the previous results 
training_targets2 = np.float32(np.zeros((len(trainx_unl_org), 250))) 
training_targets3 = np.float32(np.zeros((len(trainx_unl_org), 10))) 

ensemble_prediction = np.float32(np.zeros((len(trainx_unl_org), 10)))
ensemble_prediction2 = np.float32(np.zeros((len(trainx_unl_org), 250)))
ensemble_prediction3 = np.float32(np.zeros((len(trainx_unl_org), 10))) 


training_target_var = np.float32(np.zeros((args.batch_size, 10)))
training_target_var2 = np.float32(np.zeros((args.batch_size, 250)))
training_target_var3 = np.float32(np.zeros((args.batch_size, 10)))


txs_new = txs
tys_new = tys



for epoch in range(300):  # 300 epochs
    begin = time.time()
    lr = args.learning_rate   #no decay of learning rate
    # construct randomly permuted minibatches
    trainx = []
    trainy = []

    trainx_unl = []
    trainx_unl2 = []



    if epoch >= 100  and epoch%20 ==0:  # after 100 epochs, for every 20 epoch, change dataset.
        txs_new = txs
        tys_new = tys
        tempx = []
        tempy = []
        tempx.append(trainx_unl_org[np.max(training_targets3,axis = 1)>0.99]) 
        tempy.append(training_targets3[np.max(training_targets3,axis = 1)>0.99])
        tempx = np.squeeze(np.array(tempx))
        tempy = np.squeeze(np.array(tempy))
        tempy_onehot = tempy.argmax(axis = 1)
        #print(tempy.shape)
        #print(tempx.shape)
        #print(tempy_onehot[10])
        txs_new = np.concatenate((txs_new,tempx),axis = 0)
        tys_new = np.append(tys_new,np.int32(tempy_onehot))





    print(txs_new.shape)
    print(tys_new.shape)
    #print(training_targets3.shape[0])
    #print(training_targets3[0])




    for t in range(int(np.ceil(trainx_unl_org.shape[0]/float(txs_new.shape[0])))): 
        inds = rng.permutation(txs_new.shape[0])
        trainx.append(txs_new[inds])  #shuffle
        trainy.append(tys_new[inds])  #shuffle  50000 labeled! 
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0) # labeled data    
	


    indices_all = rng.permutation(trainx_unl_org.shape[0])
    trainx_unl = trainx_unl_org[indices_all]    # all can be treated as unlabeled examples
    trainx_unl2 = trainx_unl2_org[rng.permutation(trainx_unl2_org.shape[0])] # trainx_unl2 not equals to trainx_unl, the indexs are different
    training_target_var = training_targets[indices_all]
    training_target_var2 = training_targets2[indices_all]      #force the labeled and unlabeled to be the same 50000:50000	  1:1
    training_target_var3 = training_targets3[indices_all]      #force the labeled and unlabeled to be the same


    epoch_predictions = np.float32(np.zeros((len(trainx_unl_org), 10)))
    epoch_predictions2 = np.float32(np.zeros((len(trainx_unl_org), 250)))	
    epoch_predictions3 = np.float32(np.zeros((len(trainx_unl_org), 10)))
	
    training_targets = np.float32(training_targets)
    training_targets2 = np.float32(training_targets2)
    training_targets3 = np.float32(training_targets3)	

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.
    train_err2 = 0.
    gen_loss = 0.



    for t in range(nr_batches_train): 
        ran_from = t*args.batch_size
        ran_to = (t+1)*args.batch_size
        ll, lu, te,te2,prediction,prediction2,Last_Result = train_batch_disc(trainx[ran_from:ran_to],trainy[ran_from:ran_to],
                                      trainx_unl[ran_from:ran_to],training_target_var[ran_from:ran_to],training_target_var2[ran_from:ran_to],lr) 
        indices = indices_all[ran_from:ran_to]
        loss_lab += ll
        loss_unl += lu
        train_err += te
        train_err2 +=te2       
        e = train_batch_gen(trainx_unl2[t*args.batch_size:(t+1)*args.batch_size],lr)  # disc and gen for unlabeled data are different
        ##print(e)
        gen_loss += float(e)
        for i, j in enumerate(indices):
            epoch_predictions[j] = prediction[i] # Gather epoch predictions.
            epoch_predictions2[j] = prediction2[i] # Gather epoch predictions.        
            epoch_predictions3[j] = Last_Result[i] # Gather epoch Last_Result after softmax.        
    # record the results
    ensemble_prediction = (prediction_decay * ensemble_prediction) + (1.0 - prediction_decay) * epoch_predictions
    training_targets = ensemble_prediction / (1.0 - prediction_decay ** ((epoch - start_epoch) + 1.0))

    ensemble_prediction2 = (prediction_decay * ensemble_prediction2) + (1.0 - prediction_decay) * epoch_predictions2
    training_targets2 = ensemble_prediction2 / (1.0 - prediction_decay ** ((epoch - start_epoch) + 1.0))


    ensemble_prediction3 = (prediction_decay * ensemble_prediction3) + (1.0 - prediction_decay) * epoch_predictions3
    training_targets3 = ensemble_prediction3 / (1.0 - prediction_decay ** ((epoch - start_epoch) + 1.0))

    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    train_err /= nr_batches_train
    train_err2 /=nr_batches_train

    # test
    test_err = 0.
    for t in range(nr_batches_test):
        test_err += test_batch(testx[t*args.batch_size:(t+1)*args.batch_size],testy[t*args.batch_size:(t+1)*args.batch_size])
    test_err /= nr_batches_test


    # report
    print("Epoch %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, train err = %.4f, train err2 = %.4f,gen loss = %.4f,test err = %.4f" % (epoch, time.time()-begin, loss_lab, loss_unl, train_err,train_err2,gen_loss,test_err))
    sys.stdout.flush()
