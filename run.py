# main
import sys
import datahandler
import gan
import numpy as np
import os
import shutil
import metrics


SYNTHETIC_DATASETS=['small_2d_ring','2d_ring','small_2d_grid','2d_grid','small_hd','hd']
REAL_DATASETS=['mnist','small_mnist']

def load_opts_2d_ring(gan_algorithm):
  opts={}
  opts['random_seed'] = 66

  opts['dataset'] = '2d_ring'
  opts['work_dir'] = 'results_ring'

  opts['num_samples'] = 100000
  opts['tf_run_batch_size'] = 500


  # Datasets
  opts['input_normalize_sym'] = False # Normalize data to [-1, 1]

  # Generative model parameters
  opts['noise_dist'] = 'normal' # uniform, normal
  opts['latent_space_dim'] = 2


  # Optimizer parameters
  opts["gan_epoch_num"] = 500

  opts['optimizer'] = 'adam' # sgd, adam
  opts["batch_size"] = 500
  opts["d_steps"] = 1
  # opts['d_new_minibatch'] = False
  opts["g_steps"] = 1
  opts['batch_norm'] = False
  opts['n_hidden']=128

  # "manual" or number (float or int) giving the number of epochs to divide
  # the learning rate by 10 (converted into an exp decay per epoch).
  opts['decay_schedule'] = 'manual'
  opts['opt_d_learning_rate'] = 0.001
  opts['opt_g_learning_rate'] = 0.001
  opts["opt_beta1"] = 0.5

  if gan_algorithm=='AdaGAN':
    opts['adagan_num_steps']=10
    opts['opt_c_learning_rate']=0.001
    opts['c_epoch_num']=10


  return opts



def load_opts_small_2d_ring(gan_algorithm):
  opts={}
  opts['random_seed'] = 66

  opts['dataset'] = 'small_2d_ring'
  opts['work_dir'] = 'results_small_ring'

  opts['num_samples'] = 100
  opts['tf_run_batch_size'] = 5


  # Datasets
  opts['input_normalize_sym'] = False # Normalize data to [-1, 1]

  # Generative model parameters
  opts['noise_dist'] = 'normal' # uniform, normal
  opts['latent_space_dim'] = 2


  # Optimizer parameters
  opts["gan_epoch_num"] = 3

  opts['optimizer'] = 'adam' # sgd, adam
  opts["batch_size"] = 5
  opts["d_steps"] = 1
  # opts['d_new_minibatch'] = False
  opts["g_steps"] = 1
  opts['batch_norm'] = False
  opts['n_hidden']=128

  # "manual" or number (float or int) giving the number of epochs to divide
  # the learning rate by 10 (converted into an exp decay per epoch).
  opts['decay_schedule'] = 'manual'
  opts['opt_d_learning_rate'] = 0.001
  opts['opt_g_learning_rate'] = 0.001
  opts["opt_beta1"] = 0.5


  if gan_algorithm=='AdaGAN':
    opts['adagan_num_steps']=3
    opts['opt_c_learning_rate']=0.001
    opts['c_epoch_num']=3


  return opts



def load_opts_2d_grid(gan_algorithm):
  opts={}
  opts['random_seed'] = 66

  opts['dataset'] = '2d_grid'
  opts['work_dir'] = 'results_grid'

  opts['num_samples'] = 100000
  opts['tf_run_batch_size'] = 500


  # Datasets
  opts['input_normalize_sym'] = False # Normalize data to [-1, 1]

  # Generative model parameters
  opts['noise_dist'] = 'normal' # uniform, normal
  opts['latent_space_dim'] = 2


  # Optimizer parameters
  opts["gan_epoch_num"] = 500

  opts['optimizer'] = 'adam' # sgd, adam
  opts["batch_size"] = 500
  opts["d_steps"] = 1
  # opts['d_new_minibatch'] = False
  opts["g_steps"] = 1
  opts['batch_norm'] = False
  opts['n_hidden']=128

  # "manual" or number (float or int) giving the number of epochs to divide
  # the learning rate by 10 (converted into an exp decay per epoch).
  opts['decay_schedule'] = 'manual'
  opts['opt_d_learning_rate'] = 0.001
  opts['opt_g_learning_rate'] = 0.001
  opts["opt_beta1"] = 0.5


  if gan_algorithm=='AdaGAN':
    opts['adagan_num_steps']=10
    opts['opt_c_learning_rate']=0.001
    opts['c_epoch_num']=10


  return opts


def load_opts_small_2d_grid(gan_algorithm):
  opts={}
  opts['random_seed'] = 66

  opts['dataset'] = 'small_2d_grid'
  opts['work_dir'] = 'results_small_grid'

  opts['num_samples'] = 100
  opts['tf_run_batch_size'] = 5


  # Datasets
  opts['input_normalize_sym'] = False # Normalize data to [-1, 1]

  # Generative model parameters
  opts['noise_dist'] = 'normal' # uniform, normal
  opts['latent_space_dim'] = 2


  # Optimizer parameters
  opts["gan_epoch_num"] = 3

  opts['optimizer'] = 'adam' # sgd, adam
  opts["batch_size"] = 5
  opts["d_steps"] = 1
  # opts['d_new_minibatch'] = False
  opts["g_steps"] = 1
  opts['batch_norm'] = False
  opts['n_hidden']=128

  # "manual" or number (float or int) giving the number of epochs to divide
  # the learning rate by 10 (converted into an exp decay per epoch).
  opts['decay_schedule'] = 'manual'
  opts['opt_d_learning_rate'] = 0.001
  opts['opt_g_learning_rate'] = 0.001
  opts["opt_beta1"] = 0.5


  if gan_algorithm=='AdaGAN':
    opts['adagan_num_steps']=3
    opts['opt_c_learning_rate']=0.001
    opts['c_epoch_num']=3


  return opts


def load_opts_hd(gan_algorithm):
  opts={}
  opts['random_seed'] = 66

  opts['dataset'] = 'hd'
  opts['work_dir'] = 'results_hd'

  opts['num_samples'] = 100000
  opts['tf_run_batch_size'] = 500


  # Datasets
  opts['input_normalize_sym'] = False # Normalize data to [-1, 1]

  # Generative model parameters
  opts['noise_dist'] = 'normal' # uniform, normal
  opts['latent_space_dim'] = 100


  # Optimizer parameters
  opts["gan_epoch_num"] = 500

  opts['optimizer'] = 'adam' # sgd, adam
  opts["batch_size"] = 500
  opts["d_steps"] = 1
  # opts['d_new_minibatch'] = False
  opts["g_steps"] = 1
  opts['batch_norm'] = False
  opts['n_hidden']=128

  # "manual" or number (float or int) giving the number of epochs to divide
  # the learning rate by 10 (converted into an exp decay per epoch).
  opts['decay_schedule'] = 'manual'
  opts['opt_d_learning_rate'] = 0.001
  opts['opt_g_learning_rate'] = 0.001
  opts["opt_beta1"] = 0.5


  if gan_algorithm=='AdaGAN':
    opts['adagan_num_steps']=10
    opts['opt_c_learning_rate']=0.001
    opts['c_epoch_num']=10


  return opts

def load_opts_small_hd(gan_algorithm):
  opts={}
  opts['random_seed'] = 66

  opts['dataset'] = 'small_hd'
  opts['work_dir'] = 'results_small_hd'

  opts['num_samples'] = 100
  opts['tf_run_batch_size'] = 5


  # Datasets
  opts['input_normalize_sym'] = False # Normalize data to [-1, 1]

  # Generative model parameters
  opts['noise_dist'] = 'normal' # uniform, normal
  opts['latent_space_dim'] = 100


  # Optimizer parameters
  opts["gan_epoch_num"] = 3

  opts['optimizer'] = 'adam' # sgd, adam
  opts["batch_size"] = 5
  opts["d_steps"] = 1
  # opts['d_new_minibatch'] = False
  opts["g_steps"] = 1
  opts['batch_norm'] = False
  opts['n_hidden']=128

  # "manual" or number (float or int) giving the number of epochs to divide
  # the learning rate by 10 (converted into an exp decay per epoch).
  opts['decay_schedule'] = 'manual'
  opts['opt_d_learning_rate'] = 0.001
  opts['opt_g_learning_rate'] = 0.001
  opts["opt_beta1"] = 0.5


  if gan_algorithm=='AdaGAN':
    opts['adagan_num_steps']=3
    opts['opt_c_learning_rate']=0.001
    opts['c_epoch_num']=10

  return opts





def load_opts_mnist(gan_algorithm):
  opts={}
  opts['random_seed'] = 66

  opts['dataset'] = 'mnist'
  opts['work_dir'] = 'results_mnist'

  opts['num_samples'] = 10000
  opts['tf_run_batch_size'] = 100


  # Datasets
  opts['input_normalize_sym'] = False # Normalize data to [-1, 1]

  # Generative model parameters
  opts['noise_dist'] = 'uniform' # uniform, normal
  opts['latent_space_dim'] = 100


  # Optimizer parameters
  opts["gan_epoch_num"] = 200

  opts['optimizer'] = 'adam' # sgd, adam
  opts["batch_size"] = 100
  opts["d_steps"] = 1
  # opts['d_new_minibatch'] = False
  opts["g_steps"] = 2
  opts['batch_norm'] = False
  opts['n_hidden']=128

  # "manual" or number (float or int) giving the number of epochs to divide
  # the learning rate by 10 (converted into an exp decay per epoch).
  opts['decay_schedule'] = 'manual'
  opts['opt_d_learning_rate'] = 0.001
  opts['opt_g_learning_rate'] = 0.005
  opts["opt_beta1"] = 0.5


  if gan_algorithm=='AdaGAN':
    opts['adagan_num_steps']=10
    opts['opt_c_learning_rate']=0.0001
    opts['c_epoch_num']=10

  if gan_algorithm=='WGAN':
    # override some arguments
    opts["d_steps"] = 5

  return opts


def load_opts_small_mnist(gan_algorithm):
  opts={}
  opts['random_seed'] = 66

  opts['dataset'] = 'small_mnist'
  opts['work_dir'] = 'results_small_mnist'

  opts['num_samples'] = 100
  opts['tf_run_batch_size'] = 5


  # Datasets
  opts['input_normalize_sym'] = False # Normalize data to [-1, 1]

  # Generative model parameters
  opts['noise_dist'] = 'uniform' # uniform, normal
  opts['latent_space_dim'] = 100


  # Optimizer parameters
  opts["gan_epoch_num"] = 3

  opts['optimizer'] = 'adam' # sgd, adam
  opts["batch_size"] = 5
  opts["d_steps"] = 1
  # opts['d_new_minibatch'] = False
  opts["g_steps"] = 2
  opts['batch_norm'] = False
  opts['n_hidden']=128

  # "manual" or number (float or int) giving the number of epochs to divide
  # the learning rate by 10 (converted into an exp decay per epoch).
  opts['decay_schedule'] = 'manual'
  opts['opt_d_learning_rate'] = 0.001
  opts['opt_g_learning_rate'] = 0.005
  opts["opt_beta1"] = 0.5


  if gan_algorithm=='AdaGAN':
    opts['adagan_num_steps']=3
    opts['opt_c_learning_rate']=0.0001
    opts['c_epoch_num']=3

  if gan_algorithm=='WGAN':
    # override some arguments
    opts["d_steps"] = 5
    
  return opts

"""
Loads training params e.g., learning rates, batch size relevant to the experiments per dataset, gan algorithm.

dataset_name: str [2d_ring, 2d_grid, hd, mnist]
gan_algorithm: str [WGAN, VEEGAN, UnrolledGAN, AdaGAN]
"""
def load_opts(dataset_name,gan_algorithm):
  if dataset_name=='2d_ring':
    opts=load_opts_2d_ring(gan_algorithm)
  elif dataset_name=='small_2d_ring':
    opts=load_opts_small_2d_ring(gan_algorithm)
  elif dataset_name=='2d_grid':
    opts=load_opts_2d_grid(gan_algorithm)
  elif dataset_name=='small_2d_grid':
    opts=load_opts_small_2d_grid(gan_algorithm)
  elif dataset_name=='hd':
    opts=load_opts_hd(gan_algorithm)
  elif dataset_name=='small_hd':
    opts=load_opts_small_hd(gan_algorithm)
  elif dataset_name=='mnist':
    opts=load_opts_mnist(gan_algorithm)
  elif dataset_name=='small_mnist':
    opts=load_opts_small_mnist(gan_algorithm)


  return opts


def main():
  dataset_name = sys.argv[1]
  gan_algorithm = sys.argv[2]

  # loads options relevant to the dataset, gan algorithm
  opts = load_opts(dataset_name,gan_algorithm)


  if os.path.isdir(opts['work_dir']):
    shutil.rmtree(opts['work_dir'])

  os.mkdir(opts['work_dir'])

    
  # loads train, test datasets
  data = datahandler.DataHandler(opts)
  data._load_data(opts)

  gan_dict = {'AdaGAN': gan.AdaGAN, 'UnrolledGAN': gan.UnrolledGAN, 'WGAN': gan.WassersteinGAN, 'VEEGAN': gan.VEEGAN}


  # closes the tf graph
  with gan_dict[gan_algorithm](opts,data) as g:
    # trains the GAN
    g.train()
  
    # sample 
    samples=g._sample_internal()

    # save samples and loss plots
    np.save(os.path.join(opts['work_dir'],'samples'),samples)
    np.save(os.path.join(opts['work_dir'],'epoch_g_loss'),g._epoch_g_loss)
    np.save(os.path.join(opts['work_dir'],'epoch_d_loss'),g._epoch_d_loss)


  # compute some metrics
  """
  # Wasserstein distance
  metrics.wass_dist(dataset_name,samples)


  # run birthday paradox metric
  # support size is the square of the s value identified by this test
  real_data = data.test_data
  s_list = [5,8,10,15]
  real_dataset_name = 'real_'+dataset_name
  metrics.birthday_paradox(real_dataset_name, real_data, s_list, opts['work_dir'])
  metrics.birthday_paradox(dataset_name, real_data, s_list, opts['work_dir'])
  """



if __name__ == "__main__":
    main()
