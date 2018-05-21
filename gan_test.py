import logging
import tensorflow.contrib.layers as layers
import tensorflow as tf
import numpy as np
st = tf.contrib.bayesflow.stochastic_tensor
import os
import shutil
import glob

"""
code citations:
-https://github.com/tolstikhin/adagan

"""



SYNTHETIC_DATASETS=['small_2d_ring','2d_ring','small_2d_grid','2d_grid','small_hd','hd']
REAL_DATASETS=['mnist','small_mnist']

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

        # training parameters
        """
        self._g_lr=opts['g_lr']
        self._d_lr=None
        self._g_num_steps=None
        self._d_num_steps=None
        self._batch_size=None
        self._gan_epoch_number=None
        self._noise_distribution=None
        self._latent_dim=None
        """
        

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
        
        
    def _generator(self,_input,is_training_ph,prefix='GENERATOR'):
        if self._opts['dataset'] in SYNTHETIC_DATASETS:
            g=self._generator_synthetic(_input,is_training_ph,prefix=prefix)
        elif self._opts['dataset'] in REAL_DATASETS:
            g=self._generator_mnist(_input,is_training_ph,prefix=prefix)

        return g


        
    
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

    def evaluate(self):
        """Train a GAN model.

        """
        with self._session.as_default(), self._session.graph.as_default():
            self._evaluate_internal()


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
                            feed_dict={self._noise_ph: batch_noise,
                                        self._is_training_ph: True})

            epoch_g_loss.append(curr_g_loss)
            epoch_d_loss.append(curr_d_loss)

    
        self._epoch_g_loss=np.array(epoch_g_loss)
        self._epoch_d_loss=np.array(epoch_d_loss)
        

        # save model - for WGAN
        print('Saving model...')
        saver.save(self._session, self._opts['work_dir']+'/checkpoint-'+str(_epoch))
        saver.export_meta_graph(self._opts['work_dir']+'/checkpoint-'+str(_epoch)+'.meta')


    def _evaluate_internal(self):
        print('Evaluating GAN')
        saver = tf.train.Saver(max_to_keep=1)
        filename = tf.train.latest_checkpoint(self._opts['work_dir'])
        print('Loading from...' + filename)
        saver.restore(self._session, filename)
        # ckpt = tf.train.get_checkpoint_state(self._opts['work_dir'])
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        # if ckpt and ckpt.model_checkpoint_path:
        #     print("Restoring previous model...")
        # try:
        #     saver.restore(self._session, ckpt.model_checkpoint_path)
        #     print("Model restored")
        # except:
        #     print("Could not restore model")
        #     pass

        # restore previous model  
        #First let's load meta graph and restore weights
        # find the right meta file
        # metafile = glob.glob(self._opts['work_dir'] + '/*.meta')[-1]
        # print(metafile)
        # saver = tf.train.import_meta_graph(metafile)
        # filename = tf.train.latest_checkpoint(self._opts['work_dir'])
        # print('Loading from...' + filename)
        # saver.restore(self._session,filename)


        # # restore the discriminator
        # graph = tf.get_default_graph()
        # op_to_restore = graph.get_tensor_by_name("Mean:0")


       
        batch_images = self._data.test_data
        print(batch_images.shape)
        fake_points = self._opts['fake_points']
        curr_d_loss=self._session.run(
                            self._d_loss,
                            feed_dict={self._real_points_ph: batch_images,
                                        self._fake_points_ph : fake_points,
                                       self._is_training_ph: False})

        print('Wasserstein distance = ' + str(np.abs(curr_d_loss)))

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
        #gen_images = self._generator(noise_ph,is_training_ph)



        d_logits_real = self._discriminator(real_points_ph,is_training_ph)

        d_logits_fake = self._discriminator(fake_points_ph, is_training_ph,reuse=True)

        # cost functions
        d_loss = tf.reduce_mean(d_logits_real - d_logits_fake)
        g_loss = tf.reduce_mean(d_logits_fake)



        self._real_points_ph = real_points_ph
        self._fake_points_ph = fake_points_ph
        self._noise_ph = noise_ph
        self._is_training_ph=is_training_ph

        self._G = None
        self._d_loss = d_loss
        self._g_loss = g_loss
        #self._c_loss = None
        #self._c_training = None
        #self._g_optim = g_optim
        #self._d_optim = d_optim
        #self._c_optim = None
        
    
      
    
        
