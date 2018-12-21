# main
import sys
import datahandler
import gan_test
import numpy as np
import os
import shutil

"""Calculates Wasserstein distance between generated and real samples."""


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

"""
fake_samples: len == size(test_data)

  size of test data for the datasets:

  grid: len(fake_samples)==100000
  ring:len(fake_samples)==100000
  hd:  ""==100000
  mnist:==10,000
"""
def evaluate_wass_dist(dataset_name, fake_samples):
  gan_algorithm = 'WGAN'
  opts = load_opts(dataset_name,gan_algorithm)
  # update work dir
  #opts['work_dir'] = '../' + opts['work_dir']
  print(opts['work_dir'])

  if not os.path.isdir(opts['work_dir']):
    print('No working directory')
    
  data = datahandler.DataHandler(opts)
  data._load_data(opts)

  # load fake samples
  #fake_samples = np.load(opts['work_dir'] + '/samples.npy')
  #print(fake_samples.shape)
  opts['fake_points'] = fake_samples


  g = gan_test.WassersteinGAN(opts, data)

  g.evaluate()



def main():

  evaluate_wass_dist('mnist', fake_samples=None)

if __name__ == "__main__":
    main()