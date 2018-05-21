"""
This code was adapted from:
https://github.com/tolstikhin/adagan

Modifications include added classes e.g., WassersteinGAN, UnrolledGAN, VEEGAN,
functions e.g., generator, discriminator.

"""



import logging
import tensorflow.contrib.layers as layers
import tensorflow as tf
import numpy as np
import os
import shutil

st = tf.contrib.bayesflow.stochastic_tensor
ds = tf.contrib.distributions
graph_replace = tf.contrib.graph_editor.graph_replace




SYNTHETIC_DATASETS=['small_2d_ring','2d_ring','small_2d_grid','2d_grid','small_hd','hd']
REAL_DATASETS=['mnist','small_mnist']

# leaky relu
def lrelu(x,leak=0.2,name='lrelu'):
    return tf.maximum(leak*x,x)

def batch_norm(_input,is_train,reuse,scope):
    return tf.contrib.layers.batch_norm(_input, center=True, scale=True,epsilon=1e-05, decay=0.9,
                is_training=is_train, reuse=reuse, updates_collections=None, scope=scope, fused=False)

class GAN(object):
    """A base class for running individual GANs.

    This class announces all the necessary bits for running individual
    GAN trainers. It is assumed that a GAN trainer should receive the
    data points and the corresponding weights, which are used for
    importance sampling of minibatches during the training. All the
    methods should be implemented in the subclasses.
    """
    def __init__(self, opts, data):

        # Create a new session with session.graph = default graph
        self._session = tf.Session()
        self._dataset=opts['dataset']
 
        
        self._data = data
        self._opts=opts
      
        # Placeholders
        self._real_points_ph = None
        self._fake_points_ph = None
        self._noise_ph = None
        self._is_training_ph=None
        # Main operations
        self._G = None

        self._d_loss = None# Loss of discriminator 
        self._g_loss = None # Loss of generator


        # Optimizers
        self._g_optim = None
        self._d_optim = None

        
        

        # saver for the graph
        # self._saver = tf.train.Saver()

        # loss
        self._epoch_g_loss=None
        self._epoch_d_loss=None



        with self._session.as_default(), self._session.graph.as_default():
            logging.debug('Building the graph...')
            self._build_model_internal(opts)
            

        # Make sure AdamOptimizer, if used in the Graph, is defined before
        # calling global_variables_initializer().
        init = tf.global_variables_initializer()
        self._session.run(init)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleaning the whole default Graph
        logging.debug('Cleaning the graph...')
        tf.reset_default_graph()
        logging.debug('Closing the session...')
        # Finishing the session
        self._session.close()
        


        
    # implements the generator architecture described in the 6.883 report based on the dataset
    def _generator(self,_input,is_training_ph,prefix='GENERATOR'):
        if self._opts['dataset'] in SYNTHETIC_DATASETS:
            g=self._generator_synthetic(_input,is_training_ph,prefix=prefix)
        elif self._opts['dataset'] in REAL_DATASETS:
            g=self._generator_mnist(_input,is_training_ph,prefix=prefix)

        return g


        
    # implements the discriminator architecture described in the 6.883 report based on the dataset
    def _discriminator(self,_input,is_training_ph,reuse=False,prefix='DISCRIMINATOR'):
        if self._opts['dataset'] in SYNTHETIC_DATASETS:
            d=self._discriminator_synthetic(_input, is_training_ph,reuse=reuse,prefix=prefix)
        elif self._opts['dataset'] in REAL_DATASETS:
            d=self._discriminator_mnist(_input, is_training_ph,reuse=reuse,prefix=prefix)

        return d
        

    def _generator_synthetic(self, _input,is_training_ph,prefix='GENERATOR'):
        input_dim = self._data.data_shape[0]
        output_shape = self._data.data_shape


        with tf.variable_scope(prefix):

            n_hidden = self._opts['n_hidden']

            layer1 = layers.fully_connected(_input, n_hidden, activation_fn=tf.nn.relu, scope = 'g_layer1')
            layer2 = layers.fully_connected(layer1, n_hidden, activation_fn=tf.nn.relu, scope = 'g_layer_2')
            out = layers.fully_connected(layer2, input_dim, activation_fn=None, scope = 'g_out') 


            out=tf.reshape(out, [-1] + list(output_shape))
        #


        return out
   

    def _generator_mnist(self, _input,is_training_ph,reuse=False,prefix="GENERATOR"):

        output_shape = self._data.data_shape

        with tf.variable_scope(prefix):
            z = layers.fully_connected(_input,  7*7*16, activation_fn=None, scope='g_z')
            z = tf.reshape(z, [-1, 7,7,16])
            z= lrelu(z)
            z= batch_norm(z,is_training_ph,reuse,scope='bn_layer1')

            conv1 = layers.convolution2d_transpose(z, 8, 5, stride=2, activation_fn=None, scope='g_conv1')
            conv1=lrelu(conv1)
            conv1=batch_norm(conv1,is_training_ph,reuse,scope='bn_layer2')

            conv2 = layers.convolution2d_transpose(conv1, 4, 5, stride=2, activation_fn=None, scope='g_conv2')
            conv2=lrelu(conv2)
            conv2=batch_norm(conv2,is_training_ph,reuse,scope='bn_layer3')

            conv3 = layers.convolution2d_transpose(conv2, 1, 5, stride=1, activation_fn=None, scope='g_conv3')
            conv3=lrelu(conv3)
            conv3=batch_norm(conv3,is_training_ph,reuse,scope='bn_layer4')

            out=tf.reshape(conv3, [-1] + list(output_shape))


        return out
         
      
    def _discriminator_synthetic(self, _input, is_training_ph,reuse=False,prefix='DISCRIMINATOR'):

        with tf.variable_scope(prefix, reuse=reuse):

            n_hidden = self._opts['n_hidden']
             
            img = layers.fully_connected(_input, n_hidden, activation_fn=tf.nn.relu, scope = 'd_layer1')
            img_flat=tf.reshape(img,[-1,128*2])
            logit = layers.fully_connected(img_flat, num_outputs=1, activation_fn=None, scope = 'd_out')


        return logit      
      
      
    def _discriminator_mnist(self, _input,is_training_ph,reuse=False,prefix='DISCRIMINATOR'):
        with tf.variable_scope(prefix, reuse=reuse):
            x = tf.reshape(_input, [-1, 28, 28,1])
            conv1 = layers.conv2d(x, 16, 5, stride=2, activation_fn=None,scope='conv1')
            conv1=lrelu(conv1)
            conv1=batch_norm(conv1,is_training_ph,reuse,scope='bn_layer1')

            conv2 = layers.conv2d(conv1, 32, 5, stride=2, activation_fn=None,scope='conv2')
            conv2=lrelu(conv2)
            conv2=batch_norm(conv2,is_training_ph,reuse,scope='bn_layer2')

            conv2=tf.reshape(conv2,[-1,7*7*32])
            log_d = layers.fully_connected(conv2, num_outputs=1, activation_fn=None,scope='fc')


        return log_d
      
      
      
      
    def train(self):
        """Train a GAN model.

        """
        with self._session.as_default(), self._session.graph.as_default():
            self._train_internal()



    def _run_batch(self, operation, placeholder, feed,
                   placeholder2=None, feed2=None):
        """Wrapper around session.run to process huge data.

        It is asumed that (a) first dimension of placeholder enumerates
        separate points, and (b) that operation is independently applied
        to every point, i.e. we can split it point-wisely and then merge
        the results. The second placeholder is meant either for is_train
        flag for batch-norm or probabilities of dropout.

        TODO: write util function which will be called both from this method
        and MNIST classification evaluation as well.

        """
        assert len(feed.shape) > 0, 'Empry feed.'
        num_points = feed.shape[0]
        batch_size = self._opts['tf_run_batch_size']
        batches_num = np.ceil((num_points + 0.) / batch_size)
        batches_num=int(batches_num)
        result = []
        for idx in range(batches_num):
            if idx == batches_num - 1:
                if feed2 is None:
                    res = self._session.run(
                        operation,
                        feed_dict={placeholder: feed[idx * batch_size:]})
                else:
                    res = self._session.run(
                        operation,
                        feed_dict={placeholder: feed[idx * batch_size:],
                                   placeholder2: feed2})
            else:
                if feed2 is None:
                    res = self._session.run(
                        operation,
                        feed_dict={placeholder: feed[idx * batch_size:
                                                     (idx + 1) * batch_size]})
                else:
                    res = self._session.run(
                        operation,
                        feed_dict={placeholder: feed[idx * batch_size:
                                                     (idx + 1) * batch_size],
                                   placeholder2: feed2})

            if len(res.shape) == 1:
                # convert (n,) vector to (n,1) array
                res = np.reshape(res, [-1, 1])
            result.append(res)
        result = np.vstack(result)
        assert len(result) == num_points
        return result

   


    def _generate_noise(self):
        if self._opts['noise_dist']=='uniform':
            noise = np.random.uniform(-1, 1, size=[self._opts['batch_size'], self._opts['latent_space_dim']]).astype(np.float32)
        elif self._opts['noise_dist']=='normal':
            noise = np.random.normal(-1, 1, size=[self._opts['batch_size'], self._opts['latent_space_dim']]).astype(np.float32)

        return noise


    # TODO
    def _build_model_internal(self,dataset):
        """Build a TensorFlow graph with all the necessary ops.

        """
        #assert False, 'Gan base class has no build_model method defined.'
        pass
    # TODO 
    def _train_internal(self):
      
        batches_num = int(self._data.num_points / self._opts['batch_size'])
        train_size = self._data.num_points

        counter = 0
        print('Training GAN')

        saver = tf.train.Saver(max_to_keep=1)

        epoch_g_loss=[]
        epoch_d_loss=[]


        for _epoch in range(self._opts["gan_epoch_num"]):
            print("epoch num" + str(_epoch))

            curr_g_loss=0
            curr_d_loss=0

            for _idx in range(batches_num):
                """
                data_ids = np.random.choice(train_size, opts['batch_size'],
                                            replace=False, p=self._data_weights)
                """
                data_ids=np.arange(self._opts['batch_size']*_idx,self._opts['batch_size']*(_idx+1)) # TODO TEST

                batch_images = self._data.data[data_ids].astype(np.float)
                
                batch_noise = self._generate_noise()
                
                # Update discriminator parameters
                for _iter in range(self._opts['d_steps']):
                    _ = self._session.run(
                        self._d_optim,
                        feed_dict={self._real_points_ph: batch_images,
                                   self._noise_ph: batch_noise,
                                   self._is_training_ph: True})
                # Update generator parameters
                for _iter in range(self._opts['g_steps']):
                    _ = self._session.run(
                        self._g_optim,
                        feed_dict={self._noise_ph: batch_noise,
                                   self._is_training_ph: True})
                counter += 1

                
                # evaluate over batch
                curr_d_loss+=self._session.run(
                            self._d_loss,
                            feed_dict={self._real_points_ph: batch_images,
                                       self._noise_ph: batch_noise,
                                       self._is_training_ph: True})
                curr_g_loss+=self._session.run(
                            self._g_loss,
                            feed_dict={self._real_points_ph: batch_images,
                                        self._noise_ph: batch_noise,
                                        self._is_training_ph: True})

            epoch_g_loss.append(curr_g_loss)
            epoch_d_loss.append(curr_d_loss)

    
        self._epoch_g_loss=np.array(epoch_g_loss)
        self._epoch_d_loss=np.array(epoch_d_loss)
        

        # save model - for WGAN
        print('Saving model...')
        saver.save(self._session, self._opts['work_dir']+'/checkpoint-'+str(_epoch))
        saver.export_meta_graph(self._opts['work_dir']+'/checkpoint-'+str(_epoch)+'.meta')

    """
    implement default
    for any algorithms that have their own, implement and override
    """
    def _sample_internal(self):
        """Sample from the trained GAN model.

        """
        noise = self._generate_noise()
        
        """
        sample = self._run_batch(
            self._opts, self._G, self._noise_ph, noise,
            self._is_training_ph, False)
        """
        samples=[]
        num_batch_samples=int(self._opts['num_samples']/self._opts['batch_size'])

        for i in range(num_batch_samples):
            sample = self._session.run(
                 self._G, feed_dict={self._noise_ph: noise,
                                    self._is_training_ph: False})

            sample=sample.reshape((self._opts['batch_size'],self._data.data_shape[0],self._data.data_shape[1],self._data.data_shape[2]))
            samples.extend(sample)

        samples=np.array(samples)
        return samples


        
"""
Based on: https://github.com/akashgit/VEEGAN
"""

class VEEGAN(GAN):
    def __init__(self, opts, data):
        GAN.__init__(self, opts, data)

    def _reconstructor(self, input_):
        input_dim = self._data.data_shape[0]
        n_hidden = self._opts['n_hidden']
        latent_dim = self._opts['latent_space_dim']
    
        with tf.variable_scope("reconstructor"):
            x = tf.reshape(input_, [-1] + [input_dim])
            h = layers.fully_connected(x, n_hidden, activation_fn=tf.nn.relu)
            h = layers.fully_connected(h, n_hidden, activation_fn=tf.nn.relu)
            z = layers.fully_connected(h, latent_dim, activation_fn=None)
            out = tf.reshape(z, [-1] + [latent_dim])
        return out

    def _build_model_internal(self, opts):
        data_shape = self._data.data_shape

        ##logging.debug("data shape")
        ##logging.debug(list(data_shape))

        # Placeholders
        real_points_ph = tf.placeholder(
                tf.float32, [None] + list(data_shape), name='real_points_ph')
        fake_points_ph = tf.placeholder(
                tf.float32, [None] + list(data_shape), name='fake_points_ph')
        noise_ph = tf.placeholder(
                tf.float32, [None] + [opts['latent_space_dim']], name='noise_ph')
        is_training_ph=tf.placeholder(tf.bool,name="is_train_ph")
        
        # Operations
        p_g = self._generator(noise_ph, is_training_ph)
        ## Data shape
        p_x = st.StochasticTensor(ds.Normal(p_g * tf.ones(data_shape),
                                              1 * tf.ones(data_shape)))
        q_z = self._reconstructor(real_points_ph)
        

        log_d_posterior = self._discriminator(real_points_ph, is_training_ph)
        log_d_prior = self._discriminator(p_g, is_training_ph,reuse=True)
        
        disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=log_d_posterior, labels=tf.ones_like(log_d_posterior)) 
                                   + tf.nn.sigmoid_cross_entropy_with_logits(logits=log_d_prior, labels=tf.zeros_like(log_d_prior)))
        
        recon_likelihood_prior =p_x.distribution.log_prob(real_points_ph)
        recon_likelihood = tf.reduce_sum(graph_replace(recon_likelihood_prior, {noise_ph: q_z}), [1])
        
        gen_loss = tf.reduce_mean(log_d_posterior) - tf.reduce_mean(recon_likelihood)

        t_vars = tf.trainable_variables()
        dvars = [var for var in t_vars if 'DISCRIMINATOR/' in var.name]
        pvars = [var for var in t_vars if 'GENERATOR/' in var.name]

        qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "reconstructor")

        ##opt_gen = tf.train.AdamOptimizer(2e-3, beta1=.5)
        ##opt_disc = tf.train.AdamOptimizer(2e-3, beta1=.5)
        opt_gen = tf.train.AdamOptimizer(opts['opt_g_learning_rate'], beta1 =opts["opt_beta1"])
        opt_disc = tf.train.AdamOptimizer(opts['opt_d_learning_rate'], beta1 =opts["opt_beta1"])
        
        train_gen_op = opt_gen.minimize(gen_loss, var_list=qvars+pvars)
        train_disc_op = opt_disc.minimize(disc_loss, var_list=dvars)


        self._real_points_ph = real_points_ph
        self._fake_points_ph = fake_points_ph
        self._noise_ph = noise_ph
        self._is_training_ph=is_training_ph
        
        ##self._real_points_ph = x
        ##self._fake_points_ph = fake_points_ph
        ##self._noise_ph = p_z
        ##self._is_training_ph=is_training_ph

        self._G = p_g
        self._d_loss = disc_loss
        self._g_loss = gen_loss
        ##self._c_loss = c_loss
        ##self._c_training = c_training
        
        self._g_optim = train_gen_op
        self._d_optim = train_disc_op
        ##self._c_optim = c_optim

    
"""
Adapted from:
https://github.com/tolstikhin/adagan

"""
class AdaGAN():
    def __init__(self,opts,data):
        self._opts=opts
        self._data=data
        self._mixture_weights=None
        self._data_weights=[] # maintain history of weights over the steps
        self._steps_made=0
        self._epoch_g_loss=[] #maintain history of epoch losses across component gans
        self._epoch_d_loss=[]

        path=os.path.join(self._opts['work_dir'],'adagan')
        if os.path.isdir(path):
            shutil.rmtree(path)

        os.mkdir(path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self



    def train(self):

        
        data_weights=np.ones((self._data.data.shape[0]))*1.0/(self._data.data.shape[0])

        for i in range(self._opts['adagan_num_steps']):
            print("adagan step %d" %i)

            
            # train component gan
            with AdaGANComponent(data_weights,self._opts,self._data) as curr_comp_gan:
                # update the current mixture weights
                beta = self._next_mixture_weight()
                # update weights over traininig set based on the previous trained gan, current beta
                if i>0:
                    data_weights=self._update_data_weights(curr_comp_gan, beta)

                self._data_weights.append(data_weights)

                # train the component gan on the reweighted training dataset
                curr_comp_gan.train()

                # keep track of stats and samples on this trained GAN
                self._epoch_d_loss.append(curr_comp_gan._epoch_d_loss)
                self._epoch_g_loss.append(curr_comp_gan._epoch_g_loss)

                sample = curr_comp_gan._sample_internal()

                np.save(os.path.join(self._opts['work_dir'],'adagan/component_{:02d}_samples.npy'.format(i)),sample)
                np.save(os.path.join(self._opts['work_dir'],'adagan/weights{:02d}.npy'.format(i)), data_weights)

            # update the mixture weights so that each component gan contributes uniformly to the mixture GAN
            if i == 0:
                self._mixture_weights = np.array([beta])
            else:
                scaled_old_weights = [v * (1.0 - beta) for v in self._mixture_weights]
                self._mixture_weights = np.array(scaled_old_weights + [beta])

            self._steps_made += 1




    """
    Uniform weighting over all compoentn GANs
    """
    def _next_mixture_weight(self):
        """Returns a weight, corresponding to the next mixture component.

        """
        return 1./(self._steps_made + 1)
            

    def _update_data_weights(self, gan, beta):
        """Update the weights of data points based on the current mixture.

        This function defines a discrete distribution over the training points
        which will be used by GAN while sampling mini batches. For AdaGAN
        algorithm we have several heuristics, including the one based on
        the theory provided in 'AdaGAN: Boosting Generative Models'.
        """
        # 1. First we need to train the big classifier, separating true data
        # from the fake one sampled from the current mixture generator.
        # Its outputs are already normalized in [0,1] with sigmoid
        prob_real_data = self._get_prob_real_data(self._opts, gan)
        prob_real_data = prob_real_data.flatten()
        
        density_ratios = (1. - prob_real_data) / (prob_real_data + 1e-8)
        data_weights = self._compute_data_weights_theory_star(beta,density_ratios)

        return data_weights


    def _get_prob_real_data(self, opts, gan):
        """Train a classifier, separating true data from the current mixture.

        Returns:
            (data.num_points,) NumPy array, containing probabilities of true
            data. I.e., output of the sigmoid function.
        """
        num_fake_images = self._data.num_points
        fake_images = self._sample_internal(num_fake_images)
        if len(self._data.data_shape)==3:
            fake_images=np.reshape(fake_images,(fake_images.shape[0],self._data.data_shape[0], self._data.data_shape[1], self._data.data_shape[2]))
        else: #assume just 2 dims
            fake_images=np.reshape(fake_images,(fake_images.shape[0],self._data.data_shape[0], self._data.data_shape[1]))
        prob_real, prob_fake = gan.train_mixture_discriminator(self._opts, fake_images) 

        return prob_real


    def _compute_data_weights_theory_star(self, beta, ratios):
        """Theory-inspired reweighting of training points.

        Refer to Section 3.1 of the arxiv paper
        """
        num = self._data.data.shape[0]
        ratios_sorted = np.sort(ratios)
        cumsum_ratios = np.cumsum(ratios_sorted)
        is_found = False
        # We first find the optimal lambda* which is guaranteed to exits.
        # While Lemma 5 guarantees that lambda* <= 1, in practice this may
        # not be the case, as we replace dPmodel/dPdata by (1-D)/D.
        for i in range(num):
            # Computing lambda from equation (18) of the arxiv paper
            _lambda = beta * num * (1. + (1.-beta) / beta \
                    / num * cumsum_ratios[i]) / (i + 1.)
            if i == num - 1:
                if _lambda >= (1. - beta) * ratios_sorted[-1]:
                    is_found = True
                    break
            else:
                if _lambda <= (1 - beta) * ratios_sorted[i + 1] \
                        and _lambda >= (1 - beta) * ratios_sorted[i]:
                    is_found = True
                    break
        # Next we compute the actual weights using equation (17)
        data_weights = np.zeros(num)
        if is_found:
            _lambdamask = ratios <= (_lambda / (1.-beta))
            data_weights[_lambdamask] = (_lambda -
                                         (1-beta)*ratios[_lambdamask]) / num / beta
            logging.debug(
                'Lambda={}, sum={}, deleted points={}'.format(
                    _lambda,
                    np.sum(data_weights),
                    1.0 * (num - sum(_lambdamask)) / num))
            # This is a delicate moment. Ratios are supposed to be
            # dPmodel/dPdata. However, we are using a heuristic
            # esplained around (16) in the arXiv paper. So the
            # resulting weights do not necessarily need to some
            # to one.
            data_weights = data_weights / np.sum(data_weights)
            return data_weights
        else:
            logging.debug(
                '[WARNING] Lambda search failed, passing uniform weights')
            data_weights = np.ones(num) / (num + 0.)
            return data_weights

    def _sample_internal(self,num_points=None):
        """Sample num elements from the current AdaGAN mixture of generators.

        In this code we are not storing individual TensorFlow graphs
        corresponding to every one of the already trained component generators.
        Instead, we sample enough of points once per every trained
        generator and store these samples. Later, in order to sample from the
        mixture, we first define which component to sample from and then
        pick points uniformly from the corresponding stored sample.

        Assumes that the component samples are saved to directory 'adagan/component_XXX_samples.npy'

        """

        #First we define how many points do we need
        #from each of the components

        # sample num points = num from the components 

        component_ids = []

        if num_points==None:
            num_points=self._opts['num_samples']
        

        for _ in xrange(num_points):
            new_id = np.random.choice(self._steps_made, 1,
                                      p=self._mixture_weights)[0]
            component_ids.append(new_id)
        points_per_component = [component_ids.count(i)
                                for i in xrange(self._steps_made)]

        # Next we sample required number of points per component
        sample = []
        for comp_id  in xrange(self._steps_made):
            _num = points_per_component[comp_id]
            if _num == 0:
                continue
            path=os.path.join(self._opts['work_dir'],'adagan','component_{:02d}_samples.npy'.format(comp_id))
            comp_samples = np.load(path)
            for _ in xrange(_num):
                sample.append(
                    comp_samples[np.random.randint(len(comp_samples))])

        # Finally we shuffle
        res = np.array(sample)
        np.random.shuffle(res)

        return res


"""
Helper class for AdaGAN to facilitate implementation and training of each component GAN.
"""

class AdaGANComponent(GAN):
    def __init__(self,data_weights,opts,data):
        self._data_weights=data_weights
        super(AdaGANComponent, self).__init__(opts,data)


    def _build_model_internal(self,opts):

        
        
        data_shape = self._data.data_shape

        # Placeholders
        real_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='real_points_ph')
        fake_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='fake_points_ph')
        noise_ph = tf.placeholder(
            tf.float32, [None] + [opts['latent_space_dim']], name='noise_ph')
        is_training_ph = tf.placeholder(tf.bool, name='is_train_ph')


        # Operations
        gen_images=self._generator(noise_ph,is_training_ph)

        d_logits_real = self._discriminator(real_points_ph, is_training_ph)
        d_logits_fake = self._discriminator(gen_images,is_training_ph, reuse=True)

        c_logits_real = self._discriminator(real_points_ph, is_training_ph, prefix='CLASSIFIER')
        c_logits_fake = self._discriminator(fake_points_ph, is_training_ph, prefix='CLASSIFIER', reuse=True)
        c_training = tf.nn.sigmoid(
            self._discriminator(real_points_ph, is_training_ph,
                               prefix='CLASSIFIER', reuse=True))

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_real, labels=tf.ones_like(d_logits_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

        c_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=c_logits_real, labels=tf.ones_like(c_logits_real)))
        c_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=c_logits_fake, labels=tf.zeros_like(c_logits_fake)))
        c_loss = c_loss_real + c_loss_fake

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'DISCRIMINATOR/' in var.name]
        g_vars = [var for var in t_vars if 'GENERATOR/' in var.name]

        # implements adam optimizer with the desired learning rates
        d_optim = tf.train.AdamOptimizer(self._opts['opt_d_learning_rate'], beta1=self._opts["opt_beta1"]).minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(self._opts['opt_g_learning_rate'], beta1=self._opts["opt_beta1"]).minimize(g_loss, var_list=g_vars)


        c_vars = [var for var in t_vars if 'CLASSIFIER/' in var.name]
        c_optim = tf.train.AdamOptimizer(self._opts['opt_c_learning_rate'], beta1=self._opts["opt_beta1"]).minimize(c_loss, var_list=c_vars)

        self._real_points_ph = real_points_ph
        self._fake_points_ph = fake_points_ph
        self._noise_ph = noise_ph
        self._is_training_ph = is_training_ph
        self._G = gen_images
        self._d_loss = d_loss
        self._g_loss = g_loss
        self._c_loss = c_loss
        self._c_training = c_training
        self._g_optim = g_optim
        self._d_optim = d_optim
        self._c_optim = c_optim

        logging.debug("Building Graph Done.")



    """
    opts: dictionary containing keys: batch size, c_epoch_num
    fake_images: ndarray of generated samples from the mixture model
    """
    def train_mixture_discriminator(self, opts, fake_images):
        """Train a classifier separating true data from points in fake_images.

        """

        batches_num = self._data.num_points / opts['batch_size']
        logging.debug('Training a mixture discriminator')

        epoch_c_loss=[]
        for epoch in range(opts["c_epoch_num"]):
            curr_c_loss=0
            for idx in range(batches_num):
                ids = np.random.choice(len(fake_images), opts['batch_size'],
                                       replace=False)
                batch_fake_images = fake_images[ids]
                ids = np.random.choice(self._data.num_points, opts['batch_size'],
                                       replace=False)
                batch_real_images = self._data.data[ids]
                _ = self._session.run(
                    self._c_optim,
                    feed_dict={self._real_points_ph: batch_real_images,
                               self._fake_points_ph: batch_fake_images,
                               self._is_training_ph: True})


                curr_c_loss+=self._session.run(
                            self._c_loss,
                            feed_dict={self._real_points_ph: batch_real_images,
                                       self._fake_points_ph: batch_fake_images,
                                       self._is_training_ph: True})
            epoch_c_loss.append(curr_c_loss)

        epoch_c_loss=np.array(epoch_c_loss)
        self._epoch_c_loss=epoch_c_loss

        """
        res = self._run_batch(
            opts, self._c_training,
            self._real_points_ph, self._data.data)
        """
        num_batches=self._data.data.shape[0]/self._opts['batch_size']
        res=[]
        for i in range(num_batches):
            data_batch=self._data.data[self._opts['batch_size']*i:self._opts['batch_size']*(i+1)]
            res_cur = self._session.run(self._c_training,feed_dict=
                                            {self._real_points_ph:data_batch,
                                            self._is_training_ph: False})
            res.extend(res_cur)
        res=np.array(res)


        num_batches=fake_images.shape[0]/self._opts['batch_size']
        res_fake=[]
        for i in range(num_batches):
            data_batch=fake_images[self._opts['batch_size']*i:self._opts['batch_size']*(i+1)]
            res_cur = self._session.run(self._c_training,feed_dict=
                                            {self._real_points_ph:data_batch,
                                            self._is_training_ph: False})
            res_fake.extend(res_cur)
        res_fake=np.array(res_fake)


        return res, res_fake


"""
Adapted from: https://github.com/tolstikhin/adagan, https://github.com/poolio/unrolled_gan
"""
class UnrolledGAN(GAN):
    def __init__(self, opts, data):
        if 'mnist' in opts['dataset']:
            self._unrolling_steps = 10
        else:
            self._unrolling_steps = 5
            
        super(UnrolledGAN, self).__init__(opts,data)
        
    def _build_model_internal(self, opts):

        data_shape = self._data.data_shape
        print("data shape", data_shape)


        # Placeholders
        real_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='real_points_ph')
        fake_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='fake_points_ph')
        noise_ph = tf.placeholder(
            tf.float32, [None] + [opts['latent_space_dim']], name='noise_ph')
        is_training_ph=tf.placeholder(tf.bool,name="is_train_ph")

        # Operations
        gen_images = self._generator(noise_ph,is_training_ph)

        d_logits_real = self._discriminator(real_points_ph,is_training_ph)
        d_logits_fake = self._discriminator(gen_images, is_training_ph,reuse=True)

        d_logits_real_cp = self._discriminator(real_points_ph, is_training_ph, reuse=False, prefix='DISCRIMINATOR_CP')
        d_logits_fake_cp = self._discriminator(gen_images, is_training_ph, reuse=True, prefix='DISCRIMINATOR_CP')

        # cost functions
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_real + d_loss_fake
        
        d_loss_real_cp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real_cp, labels=tf.ones_like(d_logits_real_cp)))
        d_loss_fake_cp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake_cp, labels=tf.zeros_like(d_logits_fake_cp)))
        d_loss_cp = d_loss_real_cp + d_loss_fake_cp

        g_loss = -d_loss_cp
        
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'DISCRIMINATOR/' in var.name]
        d_vars_cp = [var for var in t_vars if 'DISCRIMINATOR_CP/' in var.name]
        g_vars = [var for var in t_vars if 'GENERATOR/' in var.name]

        print("vars to optimize")

        # Ops to unroll back the variable values of discriminator_cp
        with tf.variable_scope('assign'):
            roll_back = []
            for var, var_cp in zip(d_vars, d_vars_cp):
                roll_back.append(tf.assign(var_cp, var))

    
        # change the optimizer to RMSProp
        learning_rate_g = opts['opt_g_learning_rate']
        learning_rate_d = opts['opt_d_learning_rate']


        d_optim = tf.train.AdamOptimizer(learning_rate_d).minimize(d_loss, var_list=d_vars)
        d_optim_cp = tf.train.AdamOptimizer(learning_rate_d).minimize(d_loss_cp, var_list=d_vars_cp)
        g_optim = tf.train.AdamOptimizer(learning_rate_g).minimize(g_loss, var_list=g_vars)
        
        
        self._real_points_ph = real_points_ph
        self._fake_points_ph = fake_points_ph
        self._noise_ph = noise_ph
        self._is_training_ph=is_training_ph

        self._G = gen_images
        self._roll_back = roll_back
        self._d_loss = d_loss
        self._d_loss_cp = d_loss_cp
        self._g_loss = g_loss
        self._g_optim = g_optim
        self._d_optim = d_optim
        self._d_optim_cp = d_optim_cp



    def _train_internal(self):
      
        batches_num = self._data.num_points / self._opts['batch_size']
        train_size = self._data.num_points

        counter = 0
        print('Training GAN')

        epoch_g_loss=[]
        epoch_d_loss=[]
        epoch_d_loss_cp=[]


        for _epoch in range(self._opts["gan_epoch_num"]):
            print("epoch num" + str(_epoch))

            curr_g_loss=0
            curr_d_loss=0
            curr_d_loss_cp=0

            for _idx in range(batches_num):
                print('Step %d of %d' % (_idx, batches_num ) )
                """
                data_ids = np.random.choice(train_size, opts['batch_size'],
                                            replace=False, p=self._data_weights)
                """
                data_ids=np.arange(self._opts['batch_size']*_idx,self._opts['batch_size']*(_idx+1)) # TODO TEST

                batch_images = self._data.data[data_ids].astype(np.float)
                
                batch_noise = self._generate_noise()
                
                # Update discriminator parameters
                for _iter in range(self._opts['d_steps']):
                    _ = self._session.run(
                        self._d_optim,
                        feed_dict={self._real_points_ph: batch_images,
                                   self._noise_ph: batch_noise,
                                   self._is_training_ph: True})
                # Unroll discriminator parameters
                for _iter in range(self._unrolling_steps):
                        _ = self._session.run(
                        self._d_optim_cp,
                        feed_dict={self._real_points_ph: batch_images,
                                   self._noise_ph: batch_noise,
                                   self._is_training_ph: True})
                # Update generator parameters
                for _iter in range(self._opts['g_steps']):
                    _ = self._session.run(
                        self._g_optim,
                        feed_dict={self._noise_ph: batch_noise,
                                   self._is_training_ph: True})
                counter += 1

                
                # evaluate over batch
                curr_d_loss+=self._session.run(
                            self._d_loss,
                            feed_dict={self._real_points_ph: batch_images,
                                       self._noise_ph: batch_noise,
                                       self._is_training_ph: True})
                curr_d_loss_cp+=self._session.run(
                            self._d_loss_cp,
                            feed_dict={self._real_points_ph: batch_images,
                                       self._noise_ph: batch_noise,
                                       self._is_training_ph: True})
                    
                curr_g_loss+=self._session.run(
                            self._g_loss,
                            feed_dict={self._noise_ph: batch_noise,
                            self._real_points_ph: batch_images,
                                        self._is_training_ph: True})

            epoch_g_loss.append(curr_g_loss)
            epoch_d_loss.append(curr_d_loss)
            epoch_d_loss_cp.append(curr_d_loss_cp)

    
        self._epoch_g_loss=np.array(epoch_g_loss)
        self._epoch_d_loss=np.array(epoch_d_loss_cp)    #store unrolled loss as attribute

        # save model - for WGAN loss
        saver = tf.train.Saver(max_to_keep=1)
        print('Saving model...')
        saver.save(self._session, self._opts['work_dir']+'/checkpoint-'+str(_epoch))
        saver.export_meta_graph(self._opts['work_dir']+'/checkpoint-'+str(_epoch)+'.meta')
    




  
"""
adapted from https://github.com/cameronfabbri/Wasserstein-GAN-Tensorflow
"""
class WassersteinGAN(GAN):
    def __init__(self, opts, data):
        GAN.__init__(self, opts, data)

    def _build_model_internal(self, opts):

        data_shape = self._data.data_shape

        #logging.debug("data shape")
        #logging.debug(list(data_shape))

        # Placeholders
        real_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='real_points_ph')
        fake_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='fake_points_ph')
        noise_ph = tf.placeholder(
            tf.float32, [None] + [opts['latent_space_dim']], name='noise_ph')
        is_training_ph=tf.placeholder(tf.bool,name="is_train_ph")

        # Operations
        gen_images = self._generator(noise_ph,is_training_ph)



        d_logits_real = self._discriminator(real_points_ph,is_training_ph)

        d_logits_fake = self._discriminator(gen_images, is_training_ph,reuse=True)

        # cost functions
        d_loss = tf.reduce_mean(d_logits_real - d_logits_fake)
        g_loss = tf.reduce_mean(d_logits_fake)


        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'DISCRIMINATOR/' in var.name]
        g_vars = [var for var in t_vars if 'GENERATOR/' in var.name]


        # specific to wasserstein loss - need to make sure function is approximately lipshitz
        clip_values = [-0.01, 0.01]
        clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for var in d_vars]
    
        # change the optimizer to RMSProp
        learning_rate_g = opts['opt_g_learning_rate']
        learning_rate_d = opts['opt_d_learning_rate']

        g_optim = tf.train.RMSPropOptimizer(learning_rate_g).minimize(g_loss, var_list=g_vars)
        d_optim = tf.train.RMSPropOptimizer(learning_rate_d).minimize(d_loss, var_list=d_vars)

        self._real_points_ph = real_points_ph
        self._fake_points_ph = fake_points_ph
        self._noise_ph = noise_ph
        self._is_training_ph=is_training_ph

        self._G = gen_images
        self._d_loss = d_loss
        self._g_loss = g_loss
        #self._c_loss = None
        #self._c_training = None
        self._g_optim = g_optim
        self._d_optim = d_optim
        #self._c_optim = None
        
    
      
    
        
