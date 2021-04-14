import tensorflow as tf
import numpy as np
from scipy import stats
from scipy.optimize import fmin_l_bfgs_b
from scipy import integrate
import matplotlib.pyplot as plt
import pdb
import time
from functools import partial
import sys


########################################
# specifying parameters of the HMC Markov Chain
########################################
mass = 1.
L = 25# 25 # no. of leapfrog steps
eps = 5.e-4 #0.025 # step size in leapfrog
N_steps = 1000 # no. of steps in the markov chain

########################################
# specifying parameters of the prior
########################################
prior_var_scale = 1.e3


def get_data():
    
    seed = 1
    np.random.seed(seed)
    tf.set_random_seed(seed)

    datafile = 'simon_results_bimodal/bim_data.txt'
    data = np.loadtxt(datafile).astype(np.float64)

    #pdb.set_trace()
    X = data[:,:-1] 
    y = data[:,-1]
    data_size = X.shape[0]

    ratio_train = 0.9
    size_train = int(np.round(data_size* ratio_train))

    perm = np.loadtxt('simon_results_bimodal/permutations_bim_data.txt',
                      delimiter=",", dtype=int)
    permutation = perm[0]
    #pdb.set_trace()
    index_train = permutation[ 0 : size_train ]
    index_test = permutation[ size_train : ]

    X_train = X[ index_train, : ]
    y_train = np.vstack(y[ index_train ])

    X_test = X[ index_test, : ]
    y_test = np.vstack(y[ index_test ])

    #Normalize the input values
    meanXTrain = np.mean(X_train, axis = 0)
    stdXTrain = np.std(X_train, axis = 0)

    meanyTrain = np.mean(y_train)
    stdyTrain = np.std(y_train)

    X_train = (X_train - meanXTrain) / stdXTrain
    X_test = (X_test - meanXTrain) / stdXTrain
    y_train = (y_train - meanyTrain) / stdyTrain

    return X_train, y_train, X_test, y_test


def w_variable(shape):
    initial = tf.random_normal(shape=shape, mean=0.0, stddev=1.e-3)
    return tf.Variable(initial)

def bias_constant(x, shape):
    return tf.constant(x, shape=shape)

def generate_initial_weights(bnn_structure, import_bias = False):
    '''
    get some initial point in weight space
    '''                                            
    W1  = w_variable([ bnn_structure[0], bnn_structure[1] ])
    W2  = w_variable([ bnn_structure[1], bnn_structure[2] ])
    W3  = w_variable([ bnn_structure[2], bnn_structure[3] ])

    if(import_bias):
        f1 = 'simon_results_bimodal//bias1_final.npy'
        f2 = 'simon_results_bimodal//bias2_final.npy'
        f3 = 'simon_results_bimodal//bias3_final.npy'
        bias1 = tf.constant(np.load(f1), dtype=tf.float32)
        bias2 = tf.constant(np.load(f2), dtype=tf.float32)
        bias3 = tf.constant(np.load(f3), dtype=tf.float32)

    else:
        bias1 = w_variable([bnn_structure[1] ])
        bias2 = w_variable([bnn_structure[2] ])
        bias3 = w_variable([bnn_structure[3] ])

    return W1, W2, W3, bias1, bias2, bias3



def create_network_output_expr(X_ph, W1, W2, W3, b1, b2, b3):
    
    # getting network output
    A1 = tf.matmul(X_ph, W1) +  b1
    h1 = tf.nn.leaky_relu(A1)
    A2 = tf.matmul(h1, W2) + b2
    h2 = tf.nn.leaky_relu(A2)
    A3 = tf.matmul(h2, W3) + b3
    y_pred = tf.reshape(A3,[-1])
    return y_pred

def output_eval(output_expr, X_ph, X,
                W1, W2, W3, W1_np, W2_np, W3_np,
                b1, b2, b3, b1_np, b2_np, b3_np, sess):
    return sess.run(output_expr, feed_dict={X_ph:X, W1:W1_np, W2:W2_np, W3:W3_np,
                                            b1:b1_np, b2:b2_np, b3:b3_np})


def create_neg_loglikelihood_expr(X_ph, y_ph, var_data, W1, W2, W3, b1, b2, b3):

    
    # getting network output
    A1 = tf.matmul(X_ph, W1) +  b1
    h1 = tf.nn.leaky_relu(A1)
    A2 = tf.matmul(h1, W2) + b2
    h2 = tf.nn.leaky_relu(A2)
    A3 = tf.matmul(h2, W3) + b3
    y_pred = tf.reshape(A3,[-1])

    ### building the likelihood
    #############################    
    N = tf.shape(y_ph)[0] # no. of points

    log_likelihood =  -tf.reduce_sum( (tf.reshape(y_ph,[-1]) - y_pred)**2/(2.*var_data) )
    log_likelihood += -tf.cast(N, tf.float32) * tf.cast(tf.log(tf.sqrt(2.*np.pi*var_data)), tf.float32)

    return -log_likelihood 

def negLogLike_eval(negloglike_expr, X_ph, X, y_ph, y,
                    W1, W2, W3, W1_np, W2_np, W3_np, 
                    b1, b2, b3, b1_np, b2_np, b3_np, sess):    
    return sess.run(negloglike_expr, feed_dict={X_ph: X, y_ph: y,
                                                W1: W1_np, W2: W2_np, W3: W3_np,
                                                b1: b1_np, b2: b2_np, b3: b3_np})


def create_neg_logjoint_expr(X_ph, y_ph, var_data, W1, W2, W3, b1, b2, b3):

    
    # getting network output
    A1 = tf.matmul(X_ph, W1) +  b1
    h1 = tf.nn.leaky_relu(A1)
    A2 = tf.matmul(h1, W2) + b2
    h2 = tf.nn.leaky_relu(A2)
    A3 = tf.matmul(h2, W3) + b3
    y_pred = tf.reshape(A3,[-1])


    ### building the prior
    ###############################
    # setting the prior ad hoc, Gaussian with Cov = Id and means = 0
    # assuming independence of all weight parameters
    f1m = 'simon_results_bimodal//W1_mean_prior.npy'
    f1l = 'simon_results_bimodal//W1_logvar_prior.npy'
    f2m = 'simon_results_bimodal//W2_mean_prior.npy'
    f2l = 'simon_results_bimodal//W2_logvar_prior.npy'
    f3m = 'simon_results_bimodal//W3_mean_prior.npy'
    f3l = 'simon_results_bimodal//W3_logvar_prior.npy'

    #means_W1 = tf.zeros(W1.shape, tf.float32)
    means_W1 = tf.constant(np.load(f1m), dtype=tf.float32) 
    means_W2 = tf.constant(np.load(f2m), dtype=tf.float32)
    means_W3 = tf.constant(np.load(f3m), dtype=tf.float32)
    #variance_elements_W1 = prior_var_scale * tf.ones(W1.shape, tf.float32)
    logvar_1 = tf.constant(np.load(f1l), dtype=tf.float32)
    variance_elements_W1 = tf.exp(logvar_1)
    logvar_2 = tf.constant(np.load(f2l), dtype=tf.float32)
    variance_elements_W2 = tf.exp(logvar_2)
    logvar_3 = tf.constant(np.load(f3l), dtype=tf.float32)
    variance_elements_W3 = tf.exp(logvar_3)

    inv_variance_elements_W1 = tf.reciprocal(variance_elements_W1)
    inv_variance_elements_W2 = tf.reciprocal(variance_elements_W2)
    inv_variance_elements_W3 = tf.reciprocal(variance_elements_W3)
    
    logprior_arr1 = -0.5*tf.multiply((W1-means_W1)**2, inv_variance_elements_W1)
    logprior_arr1 += -tf.log(tf.sqrt(2.*np.pi * variance_elements_W1))
    logprior_arr2 = -0.5*tf.multiply((W2-means_W2)**2, inv_variance_elements_W2)
    logprior_arr2 += -tf.log(tf.sqrt(2.*np.pi * variance_elements_W2))
    logprior_arr3 = -0.5*tf.multiply((W3-means_W3)**2, inv_variance_elements_W3)
    logprior_arr3 += -tf.log(tf.sqrt(2.*np.pi * variance_elements_W3))
    #pdb.set_trace()
    log_prior = tf.reduce_sum(logprior_arr1) + tf.reduce_sum(logprior_arr2) \
                + tf.reduce_sum(logprior_arr3) 
    

    ### building the likelihood
    #############################    
    N = tf.shape(y_ph)[0] # no. of points

    log_likelihood =  -tf.reduce_sum( (tf.reshape(y_ph, [-1]) - y_pred)**2/(2.*var_data) )

    log_likelihood += -tf.cast(N, tf.float32) * tf.cast(tf.log(tf.sqrt(2.*np.pi*var_data)), tf.float32)

    return -log_likelihood - log_prior

def U_eval(neglogjoint_expr, X_ph, X, y_ph, y,
           W1, W2, W3, W1_np, W2_np, W3_np, b1, b2, b3, b1_np, b2_np, b3_np,
           sess):    
    return sess.run(neglogjoint_expr, feed_dict={X_ph: X, y_ph: y,
                                                 W1: W1_np, W2: W2_np, W3: W3_np,
                                                 b1: b1_np, b2: b2_np, b3:b3_np})


def create_gradU_expr(neglogjoint_expr, W1, W2, W3):
    return tf.gradients(neglogjoint_expr, [W1, W2, W3])


def gradU_eval(gradU_expr, X_ph, X, y_ph, y,
               W1, W2, W3, W1_np, W2_np, W3_np,
               sess):
    grads = sess.run(gradU_expr, feed_dict={X_ph: X, y_ph: y,
                                            W1: W1_np, W2: W2_np, W3: W3_np})
    
    grads_1d = np.hstack((grads[0].reshape((-1)),
                          grads[1].reshape((-1)),
                          grads[2].reshape((-1)) ))
    
    return grads_1d


def HMC_1step(mass, eps, L, current_q, biases, neglogjoint_hmc, 
              grad_neglogjoint_hmc):

    q = current_q
    q_history = []
    q_history.extend(q)
    
    current_U = neglogjoint_hmc(current_q, biases)
    #pdb.set_trace()
    p_cov = mass*np.eye(len(q))
    p_mean = np.zeros(len(q))
    p = np.random.multivariate_normal(p_mean, p_cov)
    #p = np.random.standard_normal(len(q))
    current_p = p

    current_K = 0.5*np.sum(current_p**2)


    p = p - 0.5*eps*grad_neglogjoint_hmc(current_q)

    #pdb.set_trace()
    for i in range(1,L+1):
        t0 = time.time()
        q = q + eps*p/mass
        #pdb.set_trace()
        q_history.extend(q)

        if(i != L):
            p = p - eps*grad_neglogjoint_hmc(q)
        #print('leapfrog i, max(q), time =',i,np.max(q), time.time()-t0)
            
    q_history = np.array(q_history).reshape((-1, len(q) ))
    
    p = p - 0.5*eps*grad_neglogjoint_hmc(q)
    p = -p
    

    proposed_U = neglogjoint_hmc(q, biases)
    proposed_K = 0.5*np.sum(p**2)

    log_prob = np.minimum(0., -proposed_U + current_U - proposed_K + current_K )
    print('log_prob=',log_prob)
    log_coin_toss = np.log(np.random.rand())
    if(log_coin_toss < log_prob):
        print('accepted')
        label = 1
        return label, q, q_history
    else:
        label = 0
        print('rejected')
        return label, current_q, q_history


def HMC_chain(mass, eps, L, N_steps, current_q, biases, neglogjoint_hmc,
              grad_neglogjoint_hmc):
    
    q = current_q
    chain = []
    chain.extend(current_q)

    acc_rate_arr=[]
    acceptance_rate = 0
    for i in range(N_steps):
        start = time.time()
        label, q, q_history = HMC_1step(mass, eps, L, q, biases, neglogjoint_hmc,
                                        grad_neglogjoint_hmc)
        acceptance_rate += label
        acc_rate_arr.append(label)
        chain.extend(q)
        print('chain step # ',i, 'exec time=',time.time()-start)

    acceptance_rate = acceptance_rate/N_steps 
    chain = np.array(chain).reshape((-1, len(q)))
    return acceptance_rate, acc_rate_arr, chain
    


def main(task):

    task1 = False # test 1 HMC point
    task2 = False # run the markov chain
    task3 = False # predictions and plot

    if task=='1':
        task1 = True
    elif task=='2':
        task2 = True
    else:
        task3 = True


    # getting data
    X_train, y_train, X_test, y_test = get_data()
    
    
    X, y = X_train, y_train
    log_var = np.load('simon_results_bimodal/log_sigma2_noise_final.npy')
    variance = np.exp(log_var)

    #pdb.set_trace()
    # defining placeholders for the data
    X_ph = tf.placeholder(tf.float32, [None, X.shape[1]])
    y_ph = tf.placeholder(tf.float32, None)
    
    # getting initial weights 
    D = X.shape[1] # dim of input
    H1 = 50
    H2 = 50
    O = 1
    bnn_structure = [D,H1,H2,O] # input layer, hidden layer, output layer
    W1, W2, W3, b1, b2, b3 = generate_initial_weights(bnn_structure, 
                                                      import_bias=True)

    ## create the neg log joint expr. once and for all
    neglogjoint_expr = create_neg_logjoint_expr(X_ph, y_ph, variance,
                                                W1, W2, W3, b1, b2, b3)
    
    ## the same for the expr. of the gradient
    gradU_expr = create_gradU_expr(neglogjoint_expr, W1, W2, W3)

    ## create expr. for the neg log likelihood, for checking purposes
    negloglike_expr = create_neg_loglikelihood_expr(X_ph, y_ph, variance,
                                                     W1, W2, W3, b1, b2, b3)

    ## create expr. for the network output, for making predictions
    network_output_expr = create_network_output_expr(X_ph, W1, W2, W3, 
                                                     b1, b2, b3)

    
    #############################################
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    #############################################
    
    # numpy 1D array for the initial weights
    w_init = np.concatenate((tf.reshape(W1, [W1.shape[0]*W1.shape[1]]).eval(),
                             tf.reshape(W2, [W2.shape[0]*W2.shape[1]]).eval(),
                             tf.reshape(W3, [W3.shape[0]*W3.shape[1]]).eval()))
    
    biases = np.concatenate((b1.eval(), b2.eval(), b3.eval() )) 

    #pdb.set_trace()

    # encapsulating function for evaluating the neglogjoint
    def neglogjoint_hmc(w, biases):
        dim1 = D*H1
        dim2 = dim1 + H1*H2
        W1_np = w[:dim1].reshape((W1.shape))
        W2_np = w[dim1: dim2].reshape((W2.shape))
        W3_np = w[dim2:].reshape((W3.shape))
        b1_np = biases[:H1].reshape(b1.shape)
        b2_np = biases[H1: H1 + H2].reshape(b2.shape)
        b3_np = biases[-1].reshape(b3.shape)
        
        return U_eval(neglogjoint_expr, X_ph, X, y_ph, y,
                      W1, W2, W3, W1_np, W2_np, W3_np, 
                      b1, b2, b3, b1_np, b2_np, b3_np,
                      sess).astype(np.float64)


    # encapsulating function for evaluating the grad of the neglogjoint
    def grad_neglogjoint_hmc(w):
        dim1 = D*H1
        dim2 = dim1 + H1*H2
        W1_np = w[:dim1].reshape((W1.shape))
        W2_np = w[dim1: dim2].reshape((W2.shape))
        W3_np = w[dim2:].reshape((W3.shape))
        
        return gradU_eval(gradU_expr, X_ph, X, y_ph, y,
                          W1, W2, W3, W1_np, W2_np, W3_np,
                          sess).astype(np.float64)


    # encapsulating function for evaluating the negloglikelihood
    def negloglike_np(w,biases):
        dim1 = D*H1
        dim2 = dim1 + H1*H2
        W1_np = w[:dim1].reshape((W1.shape))
        W2_np = w[dim1: dim2].reshape((W2.shape))
        W3_np = w[dim2:].reshape((W3.shape))
        b1_np = biases[:H1].reshape(b1.shape)
        b2_np = biases[H1: H1 + H2].reshape(b2.shape)
        b3_np = biases[-1].reshape(b3.shape)
        
        return negLogLike_eval(negloglike_expr, X_ph, X, y_ph, y,
                               W1, W2, W3, W1_np, W2_np, W3_np,
                               b1, b2, b3, b1_np, b2_np, b3_np, 
                               sess)



    # encapsulating function for evaluating the network output
    def network_output(Xtest,w,biases):
        dim1 = D*H1
        dim2 = dim1 + H1*H2
        W1_np = w[:dim1].reshape((W1.shape))
        W2_np = w[dim1: dim2].reshape((W2.shape))
        W3_np = w[dim2:].reshape((W3.shape))
        b1_np = biases[:H1].reshape(b1.shape)
        b2_np = biases[H1: H1 + H2].reshape(b2.shape)
        b3_np = biases[-1].reshape(b3.shape)

        return output_eval(network_output_expr, X_ph, Xtest, 
                           W1, W2, W3, W1_np, W2_np, W3_np, 
                           b1, b2, b3, b1_np, b2_np, b3_np, 
                           sess)

    
    ###############################################
    # obtaining an optimal point by maximizing the log joint
    # this will define the t=0 point of the Markov chain
    ################################################
    
    '''
    func = partial(neglogjoint_hmc, biases)
    #grad_func = partial(grad_neglogjoint_hmc, biases)
    
    
    w_opt, fmin, etc = fmin_l_bfgs_b(func, w_init,
                                     fprime=grad_neglogjoint_hmc)
    
    pdb.set_trace()
    '''

    if(task1): 
        #################################################
        # testing 1 point of the chain
        ################################################
        start = time.time()
        res = HMC_1step(mass, eps, L, w_init, biases, neglogjoint_hmc, 
                        grad_neglogjoint_hmc)
        print('exec time = ',time.time()-start)
                    
    

    if(task2):
        #################################################
        # obtaining the chain
        ################################################
        acc_rate, acc_rate_arr, mchain = HMC_chain(mass, eps, L, N_steps, 
                                                   w_init, biases,
                                                   neglogjoint_hmc,
                                                   grad_neglogjoint_hmc)

        print('acceptance rate = ',acc_rate)
        np.save('mchain_bimodal_N'+str(N_steps)+'_eps'+str(eps)+'_L'+str(L)+'_m'+str(mass), mchain)
        #pdb.set_trace() 


    if(task3):
        ##############################
        # loading the chain
        ##############################

        arr=np.load('mchain_results_bimodal/mchain_bimodal_N'+str(N_steps)+'_eps'+str(eps)+'_L'+str(L)+'_m'+str(mass)+'.npy')

        '''
        negloglike_arr=[]
        for w in arr:
            negloglike_arr.append(negloglike_np(w,biases))
        pdb.set_trace()
        quit()
        '''
        
        arr=arr[800:]
        

        X_grid = np.linspace(np.min(X_test[:,0]),np.max(X_test[:,0]),100)
        ypred_arr=[]
        i=0
        for w in arr:
            print(i, end= " ")
            i+=1
            ypred_arr.extend([network_output(X_grid.reshape((-1,1)),w, biases)])
        print('.')
        ypred_arr=np.array(ypred_arr)
        ypred_arr=ypred_arr.reshape((ypred_arr.shape[1],ypred_arr.shape[0]))
        
        fmean = np.mean(ypred_arr,axis=1)
        fstd = np.std(ypred_arr,axis=1,ddof=1)
        #pdb.set_trace()

        y_test = (y_test - np.mean(y_test))/np.std(y_test,ddof=1)
        plt.scatter(X_test,y_test,color='gray',label='test data')
        #pdb.set_trace()
        for i in range(10):#(len(arr)-10,len(arr)): #range(ypred_arr.shape[1]):
            plt.scatter(X_grid,ypred_arr[:,i],s=1.)
        plt.xlabel('x',fontsize=12)
        plt.ylabel('y',fontsize=12)
        plt.legend()
        plt.title('output for the samples 800-810, HMC w/o bias, L=25, eps=5.e-4, N=1000')
        plt.savefig('output_800-810_bimodal_hmc_wo-bias_N=1000_L=25_eps=5.e-4.pdf')
        plt.show()
        pdb.set_trace()

        quit()
        plt.plot(X_grid,fmean, color='blue', label='mean pred.')
        plt.plot(X_grid,fmean-3*fstd,color='gray', label=r'$\pm 3\sigma$')
        plt.plot(X_grid,fmean+3*fstd,color='gray')
        #plt.ylim(-100,100)
        #plt.xlim(-6,6)
        plt.legend()
        plt.title('std data, prior variance = 1*Id')
        #plt.savefig("mchain_results_reprod_150205336/mchain_with-bias_std_priorvar1_N1000_eps0.001_L25.pdf")
        plt.show()
        quit()
        
    
    

if __name__ == '__main__':
    
    task = sys.argv[1]

    main(task)


