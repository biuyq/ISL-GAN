import argparse
import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import nn
import sys
import plotting
import scipy
import scipy.misc
import svhn_data


from scipy import linalg



# settings
factor_M = 0.0
LAMBDA_2 = 2.0
prediction_decay = 0.6   
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=2)
parser.add_argument('--seed_data', default=2)
parser.add_argument('--count', default=50)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--unlabeled_weight', type=float, default=2.0)
parser.add_argument('--learning_rate', type=float, default=0.0003)# learning rate, no decay
parser.add_argument('--data_dir', type=str, default='//home/weilegexiang/Desktop/CT-GAN-master/CT-GANs') #add your own path
args = parser.parse_args()
print(args)

# fixed random seeds
rng_data = np.random.RandomState(args.seed_data)
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))


# load 	SVHN

def rescale(mat):
    return np.transpose(np.cast[th.config.floatX]((-127.5 + mat)/127.5),(3,2,0,1))
#def rescale(mat):
#    mat = mat - np.mean(mat,axis = (1,2,3),keepdims = True)
#    mat = mat / (np.mean(mat**2,axis = (1,2,3),keepdims=True)**0.5)
#    return  np.transpose(np.cast[th.config.floatX](mat),(3,2,0,1))

trainx, trainy = svhn_data.load(args.data_dir,'train')
testx, testy = svhn_data.load(args.data_dir,'test')
trainx = rescale(trainx)
testx = rescale(testx)
nr_batches_train = int(trainx.shape[0]/args.batch_size)
nr_batches_test = int(np.ceil(float(testx.shape[0])/args.batch_size))




#######   
#pad
#######

trainx = np.pad(trainx, ((0, 0), (0, 0), (2, 2), (2, 2)), 'reflect')



trainx_unl_org = trainx.copy()
trainx_unl2_org = trainx.copy()
 


# specify generative model input with 50 dim
noise_dim = (args.batch_size, 50)
noise = theano_rng.uniform(size=noise_dim)
gen_layers = [ll.InputLayer(shape=noise_dim, input_var=noise)]
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu), g=None))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (args.batch_size,512,4,4)))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 4 -> 8
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 8 -> 16
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,3,32,32), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1)) # 16 -> 32
gen_dat = ll.get_output(gen_layers[-1])

## same as the original net  the size in tempens  128 - 256
disc_layers = [ll.InputLayer(shape=(None, 3, 32, 32))]
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.2))
#disc_layers.append(ll.GaussianNoiseLayer(disc_layers[-1], sigma=0.15))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 256, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 512, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=256, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=128, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=10, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))
disc_params = ll.get_all_params(disc_layers, trainable=True)

# costs
labels = T.ivector()
x_lab = T.tensor4()
x_unl = T.tensor4()
training_targets =T.matrix('targets')
training_targets2 = T.matrix('targets2') 
training_targets3 = T.matrix('targets3') 

temp = ll.get_output(gen_layers[-1], deterministic=False, init=True) 
temp = ll.get_output(disc_layers[-1], x_lab, deterministic=False, init=True) 
init_updates = [u for l in gen_layers+disc_layers for u in getattr(l,'init_updates',[])]

output_before_softmax_lab = ll.get_output(disc_layers[-1], x_lab, deterministic=False) # no softmax labeled dis output
output_before_softmax_unl,output_before_softmax_unl_ = ll.get_output([disc_layers[-1],disc_layers[-2]], x_unl, deterministic=False)  # last two layers' output 
output_before_softmax_gen = ll.get_output(disc_layers[-1], gen_dat, deterministic=False) #dis of generator output 

l_lab = output_before_softmax_lab[T.arange(args.batch_size),labels] 
l_unl = nn.log_sum_exp(output_before_softmax_unl) 
l_unl_ = nn.log_sum_exp(output_before_softmax_unl_) 
l_gen = nn.log_sum_exp(output_before_softmax_gen)
#loss_lab = -T.mean(l_lab) + T.mean(T.mean(nn.log_sum_exp(output_before_softmax_lab)))
loss_lab = T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(output_before_softmax_lab),labels)*(T.exp((pow(T.nnet.softmax(output_before_softmax_lab)[T.arange(args.batch_size),labels],2.0)))))
######################
#the consistency term
######################

loss_ct = T.mean(lasagne.objectives.squared_error(T.nnet.softmax(output_before_softmax_unl),T.nnet.softmax(training_targets)),axis = 1) #last layer should be with softmax,not only seperate the real from fake, but also the class of real it belongs to, D

## last line after softmax

last_result = T.nnet.softmax(output_before_softmax_unl)


loss_ct_ = T.mean(lasagne.objectives.squared_error(output_before_softmax_unl_,training_targets2),axis = 1)    #D_

CT = LAMBDA_2*(loss_ct+loss_ct_*0.2)-factor_M  # 1.0:0.2
CT_ = T.mean(T.maximum(CT,0.0*CT),axis = 0)

loss_unl = 0.5*(CT_ -T.mean(l_unl) + T.mean(T.nnet.softplus(l_unl)) -np.log(1) + T.mean(T.nnet.softplus(l_gen))) 

zeros = np.zeros(args.batch_size)
train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))
train_err2 = T.mean(T.le(T.max(output_before_softmax_lab,axis=1),zeros))  #mis-classification


# test error
output_before_softmax = ll.get_output(disc_layers[-1], x_lab, deterministic=True) # no training
test_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),labels))

# Theano functions for training the disc net
lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)


disc_param_updates = nn.adam_updates(disc_params, loss_lab + args.unlabeled_weight*loss_unl, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates) # data based initialization
train_batch_disc = th.function(inputs=[x_lab,labels,x_unl,training_targets,training_targets2,lr], outputs=[loss_lab, loss_unl, train_err,train_err2,output_before_softmax_unl,output_before_softmax_unl_,last_result], updates=disc_param_updates+disc_avg_updates)

test_batch = th.function(inputs=[x_lab], outputs=output_before_softmax, givens=disc_avg_givens)
samplefun = th.function(inputs=[],outputs=gen_dat)

# Theano functions for training the gen net
output_unl = ll.get_output(disc_layers[-2], x_unl, deterministic=False)
output_gen = ll.get_output(disc_layers[-2], gen_dat, deterministic=False)
m1 = T.mean(output_unl,axis=0)
m2 = T.mean(output_gen,axis=0)
loss_gen = T.mean(abs(m1-m2)) # feature matching loss, L1 loss
gen_params = ll.get_all_params(gen_layers, trainable=True)


gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
train_batch_gen = th.function(inputs=[x_unl,lr], outputs=loss_gen, updates=gen_param_updates)

# select labeled data
inds = rng_data.permutation(trainx.shape[0])  
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for j in range(10):  
    txs.append(trainx[trainy==j][:args.count]) 
    tys.append(trainy[trainy==j][:args.count]) 
txs = np.concatenate(txs, axis=0) 
tys = np.concatenate(tys, axis=0) 
#print(txs.shape)
# //////////// perform training //////////////



start_epoch = 0

training_targets = np.float32(np.zeros((len(trainx_unl_org), 10)))  # for saving the previous results 
training_targets2 = np.float32(np.zeros((len(trainx_unl_org), 128))) 
training_targets3 = np.float32(np.zeros((len(trainx_unl_org), 10))) 

ensemble_prediction = np.float32(np.zeros((len(trainx_unl_org), 10)))
ensemble_prediction2 = np.float32(np.zeros((len(trainx_unl_org), 128)))
ensemble_prediction3 = np.float32(np.zeros((len(trainx_unl_org), 10))) 


training_target_var = np.float32(np.zeros((args.batch_size, 10)))
training_target_var2 = np.float32(np.zeros((args.batch_size, 128)))
training_target_var3 = np.float32(np.zeros((args.batch_size, 10)))


txs_new = txs
tys_new = tys

for epoch in range(1500): #no learning rate decay. More epochs may give better result
    begin = time.time()
    lr = args.learning_rate   #no decay of learning rate
    trainx = [] #empty
    trainy = []
    trainx_unl = []
    trainx_unl2 = []


    if epoch >= 500  and epoch%20 ==0:  # after 500 epochs, for every 100 epoch, change dataset.
        txs_new = txs
        tys_new = tys
        #for  ite in range(training_targets3.shape[0]):
        #    a = training_targets3[ite]
        #    value = max(a)
        #    #print(value)
        #    if value >= 0.3: # can be changed
        #        #print(value)
        #        index_max = a.tolist().index(max(a))
        #        #a = float(np.zeros(10)).tolist()
        #        #a[index_max] = 1
        #        b = trainx_unl_org[ite][np.newaxis,:,:,:]
        #        #value = [value,np.newaxis]
        #        txs_new = np.concatenate((txs_new,b),axis = 0)
        #        tys_new = np.append(tys_new,np.int32(value))
        #        #txs_new.append(trainx_unl_org[ite])
        #        #tys_new.append(a)

        #tempx = []
        #tempy = [] # use all the data
        ##pos = np.where((np.max(training_targets3,axis = 1)<=0.99) & (np.max(training_targets3,axis = 1)>=0.90))
        #tempx.append(trainx_unl_org) 
        #tempy.append(training_targets3)
        #tempx = np.squeeze(np.array(tempx))
        #tempy = np.squeeze(np.array(tempy))
        #tempy_onehot = tempy.argmax(axis = 1)
        ##print(tempy.shape)
        ##print(tempx.shape)
        ##print(tempy_onehot[10])
        #txs_new = tempx
        #tys_new = np.int32(tempy_onehot)



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

##################
##prepair dataset
##################
	
    if epoch==0:
        print(trainx.shape)
        init_param(trainx[:1000]) # data based initialization
    print(trainx.shape)		
    indices_l = trainx.shape[0]
    indices_ul = trainx_unl.shape[0]
	#inde = np.range()
    noisy_a = []
    for start_idx in range(0,indices_l):  # from 0 to 50000
	
        img_pre = trainx[start_idx]  # trainx labeled
		
        #if np.random.uniform() > 0.5:
        #    img_pre = img_pre[:,:,::-1] # reversal
        t = 2
        crop = 2
        ofs0 = np.random.randint(-t, t + 1) + crop
        ofs1 = np.random.randint(-t, t + 1) + crop
        img_a = img_pre[:, ofs0:ofs0+32, ofs1:ofs1+32]
        noisy_a.append(img_a)
    noisy_a = np.array(noisy_a)
    trainx = noisy_a           
		
    noisy_a, noisy_b,noisy_c = [], [], []		
    for start_idx in range(0,indices_ul):  # from 0 to 50000
	
        img_pre_a = trainx_unl[start_idx] # unlabeled 
        img_pre_b = trainx_unl2[start_idx]	

		
        #if np.random.uniform() > 0.5:
        #    img_pre_a = img_pre_a[:,:,::-1] 
	#		
        #if np.random.uniform() > 0.5:
        #    img_pre_b = img_pre_b[:,:,::-1] 

        img_pre_c = img_pre_a
			
        t = 2
        crop = 2
        ofs0 = np.random.randint(-t, t + 1) + crop   ##crop
        ofs1 = np.random.randint(-t, t + 1) + crop
        img_a = img_pre_a[:, ofs0:ofs0+32, ofs1:ofs1+32]
        ofs0 = np.random.randint(-t, t + 1) + crop
        ofs1 = np.random.randint(-t, t + 1) + crop
        img_b = img_pre_b[:, ofs0:ofs0+32, ofs1:ofs1+32]
        ofs0 = np.random.randint(-t, t + 1) + crop
        ofs1 = np.random.randint(-t, t + 1) + crop
        img_c = img_pre_c[:, ofs0:ofs0+32, ofs1:ofs1+32]
        noisy_a.append(img_a)
        noisy_b.append(img_b) # maybe used in the future
        noisy_c.append(img_c) # maybe used in the future

    noisy_a = np.array(noisy_a)
    noisy_b = np.array(noisy_b)
    noisy_c = np.array(noisy_c)
    trainx_unl =  noisy_a 
    trainx_unl2 =  noisy_b 
    trainx_unl3 =  noisy_c 

    epoch_predictions = np.float32(np.zeros((len(trainx_unl_org), 10)))
    epoch_predictions2 = np.float32(np.zeros((len(trainx_unl_org), 128)))	
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

    print(trainx.shape)
    print("1")
    print(trainx_unl.shape)

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
    test_pred = np.zeros((len(testy),10), dtype=th.config.floatX)
    for t in range(nr_batches_test):
        last_ind = np.minimum((t+1)*args.batch_size, len(testy))
        first_ind = last_ind - args.batch_size
        test_pred[first_ind:last_ind] = test_batch(testx[first_ind:last_ind])
    test_err = np.mean(np.argmax(test_pred,axis=1) != testy)

    # report
    print("Epoch %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, train err = %.4f, train err2 = %.4f,gen loss = %.4f,test err = %.4f" % (epoch, time.time()-begin, loss_lab, loss_unl, train_err,train_err2,gen_loss,test_err))
    sys.stdout.flush()

    # generate samples from the model
    sample_x = samplefun()
    img_bhwc = np.transpose(sample_x[:100,], (0, 2, 3, 1))
    img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title='CIFAR10 samples')
    plotting.plt.savefig("cifar_sample_CT.png")

    # save params
    #np.savez('disc_params.npz', *[p.get_value() for p in disc_params])
    #np.savez('gen_params.npz', *[p.get_value() for p in gen_params])
