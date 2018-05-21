
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
from scipy.stats import multivariate_normal as scipy_normal
from sklearn.neighbors.kde import KernelDensity
import seaborn
import tensorflow as tf
import sys
import itertools
import wasserstein_distance
import os
import copy
seaborn.set_style("white")


"""
Plots kernel density estimate plots.

fake: ndarray of generated samples

name: XXXX.png filename to save the plot to
"""
def plot_kde_samples(fake, name):
   fig = plt.figure(figsize = ((2,2)))
   ax = fig.add_subplot(111)
   seaborn.kdeplot(fake[:,0], fake[:,1], shade=True, n_levels=30, ax=ax)
   plt.axis('off')
   ax.set_xticklabels([])
   ax.set_yticklabels([])
   plt.savefig(name)
   plt.close()



"""
Implements the Waaserstein critic metric as described in:
Borji, Ali. "Pros and Cons of GAN Evaluation Measures." arXiv preprint arXiv:1802.03446 (2018).


dataset_name: str [2d_ring,2d_grid,hd,mnist]
fake_points: ndarray of generated samples
"""
def wass_dist(dataset_name, fake_points):
    # wasserstein discriminator should have been trained for this specific dataset before evaluation, otherwise errors will occur
    print('Only compute the wasserstein distance after training WGAN model!')
    wasserstein_distance.evaluate_wass_dist(dataset_name, fake_points)



"""
# Coverage metrics on synthetic datasets. 
# Input: real_points, array of real data, 
# fake_points: array of generated data, same shape as real_points, 
# validation_fake_points: array of generated data, used for cross validation on KDE, could be set as None.

Adapted from 
https://github.com/tolstikhin/adagan
"""

def evaluate_vec(real_points, fake_points, validation_fake_points=None):
        """Compute the average log-likelihood and the Coverage metric.
        Coverage metric is defined in arXiv paper. It counts a mass of true
        data covered by the 95% quantile of the model density.
        """

        # Estimating density with KDE
        dist = fake_points[:-1] - fake_points[1:]
        dist = dist * dist
        dist = np.sqrt(np.sum(dist, axis=1))
        bandwidth = np.median(dist)
        num_real = len(real_points)
        num_fake = len(fake_points)
        if validation_fake_points is not None:
            max_score = -1000000.
            num_val = len(validation_fake_points)
            b_grid = bandwidth * (2. ** (np.arange(14) - 7.))
            for _bandwidth in b_grid:
                kde = KernelDensity(kernel='gaussian', bandwidth=_bandwidth)
                kde.fit(np.reshape(fake_points, [num_fake, -1]))
                score = np.mean(kde.score_samples(np.reshape(validation_fake_points, [num_val, -1])))
                if score > max_score:
                    # logging.debug("Updating bandwidth to %.4f"
                    #             " with likelyhood %.2f" % (_bandwidth, score))
                    bandwidth = _bandwidth
                    max_score = score
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(np.reshape(fake_points, [num_fake, -1]))

        # Computing Coverage, refer to Section 4.3 of arxiv paper
        model_log_density = kde.score_samples(np.reshape(fake_points, [num_fake, -1]))
        # np.percentaile(a, 10) returns t s.t. np.mean( a <= t ) = 0.1
        threshold = np.percentile(model_log_density, 5)
        real_points_log_density = kde.score_samples(np.reshape(real_points, [num_real, -1]))
        ratio_not_covered = np.mean(real_points_log_density <= threshold)
        
        C = 1. - ratio_not_covered
     
        return C
     

""" 
# Function used to count the proportion of high quality samples and number of detected modes. 
# Input: gen_data, array of generated samplesl; sd, standard deviance, set as 0.05 in our study; 
# center, array of centers of the modes, in our studies, these are set as:

Adapted from: https://github.com/akashgit/VEEGAN
"""


def cal_quality(gen_data, center, dataset_name, sd = 0.05):

  center_ring = np.array([[2,0], [2**0.5,2**0.5], [0,2], [-2**0.5, 2**0.5],
                   [-2,0], [-2**0.5, -2**0.5], [0,-2], [2**0.5, -2**0.5]],dtype='float32')


  center_grid = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                          range(-4, 5, 2))],dtype='float32')


  center_hd = np.array([[0 for i in range(500)] + [0 for i in range(300)],[0 for i in range(250)]+[1 for i in range(250)]+[0 for i in range(300)],[1 for i in range(250)]+[0 for i in range(250)]+[0 for i in range(300)], [1 for i in range(500)]+[0 for i in range(300)],
                   [-1 for i in range(250)]+[0 for i in range(250)]+[0 for i in range(300)], [0 for i in range(250)]+[-1 for i in range(250)]+[0 for i in range(300)],
                   [-1 for i in range(250)]+[1 for i in range(250)]+[0 for i in range(300)], [-1 for i in range(500)]+[0 for i in range(300)],
                   [1 for i in range(250)]+[-1 for i in range(250)]+[0 for i in range(300)]], dtype='float32')


  if dataset_name=="2d_ring":
      center=center_ring
  elif dataset_name=='2d_grid':
      center=center_grid
  elif dataset_name=='hd':
      center=center_hd

  sample_size, dim = np.shape(gen_data)
  num_center = len(center)
  ##print(dim)
  high_qual_num = 0
  cap_mode = num_center * [0]
  if dim == 800:
      threshold = 5 * sd * np.sqrt(500)
      center_array = np.array(center)[:,0:500]
      for i in range(sample_size):
          for j in range(num_center):
              x = gen_data[i][0:500]
              y = center_array[j]
              dist = np.sqrt(np.sum((x-y)**2))
              if dist < threshold:
                  high_qual_num += 1
                  cap_mode[j] = 1
  else:
      threshold = 3 * sd * np.sqrt(2)
      center_array = np.array(center)
      for i in range(sample_size):
          for j in range(num_center):
              x = gen_data[i]
              y = center_array[j]
              dist = np.sqrt(np.sum((x-y)**2))
              if dist < threshold:
                  high_qual_num += 1
                  cap_mode[j] = 1
  return high_qual_num/sample_size, sum(cap_mode)


"""
Computes the coverage, Adjusted coverage, % high quality samples, # modes captured
as described in the 6.883 proj report.

All metrics use the pretrained mnist model found under 'models' and are based on 
those samples that the classifier is highly confident on.

fake_points: ndarray
batch_size: int Used for evaluating metrics in batches


Code adapted from: 
https://github.com/tolstikhin/adagan
"""
def evaluate_mnist(fake_points,batch_size=10):
        
    num_fake = len(fake_points)


    l2s=None
    Qz=None

    THRESHOLD = 0.999 # for mnist classifier on confidence

    # Classifying points with pre-trained model.
    # Pre-trained classifier assumes inputs are in [0, 1.]
    # There may be many points, so we will sess.run
    # in small chunks.
    model_file='models/mnist_trainSteps_19999_yhat'

    with tf.Graph().as_default() as g:
        saver = tf.train.import_meta_graph(model_file + '.meta')
        with tf.Session().as_default() as sess:
            saver.restore(sess, model_file)
            input_ph = tf.get_collection('X_')
            assert len(input_ph) > 0, 'Failed to load pre-trained model'
            # Input placeholder
            input_ph = input_ph[0]
            dropout_keep_prob_ph = tf.get_collection('keep_prob')
            assert len(dropout_keep_prob_ph) > 0, 'Failed to load pre-trained model'
            dropout_keep_prob_ph = dropout_keep_prob_ph[0]
            trained_net = tf.get_collection('prediction')
            assert len(trained_net) > 0, 'Failed to load pre-trained model'
            # Predicted digit
            trained_net = trained_net[0]
            logits = tf.get_collection('y_hat')
            assert len(logits) > 0, 'Failed to load pre-trained model'
            # Resulting 10 logits
            logits = logits[0]
            prob_max = tf.reduce_max(tf.nn.softmax(logits),
                                     reduction_indices=[1])

            
            batches_num = int(np.ceil((num_fake + 0.) / batch_size))
            result = []
            result_probs = []
            result_is_confident = []
            thresh = THRESHOLD
            for idx in xrange(batches_num):
                end_idx = min(num_fake, (idx + 1) * batch_size)
                batch_fake = fake_points[idx * batch_size:end_idx]
                _res, prob = sess.run(
                    [trained_net, prob_max],
                    feed_dict={input_ph: batch_fake,
                               dropout_keep_prob_ph: 1.})
                result.append(_res)
                result_probs.append(prob)
                result_is_confident.append(prob > thresh)
            result = np.hstack(result)
            result_probs = np.hstack(result_probs)
            result_is_confident = np.hstack(result_is_confident)
            assert len(result) == num_fake
            assert len(result_probs) == num_fake

    


    digits = result.astype(int)
    # Plot one fake image per detected mode
    gathered = []
    points_to_plot = []
    for (idx, dig) in enumerate(list(digits)):
        if not dig in gathered and result_is_confident[idx]:
            gathered.append(dig)
            p = result_probs[idx]
            points_to_plot.append(fake_points[idx])
            #print('Mode %03d covered with prob %.3f' % (dig, p))
    # Confidence of made predictions
    conf = np.mean(result_probs)
    

    if np.sum(result_is_confident) == 0:
        C_actual = 0.
        C = 0.
        JS = 2.
    else:
        # Compute the actual coverage
        C_actual = len(np.unique(digits[result_is_confident])) / 10.
        # Compute Pdata(Pmodel > t) where Pmodel( Pmodel > t ) = 0.95
        # np.percentaile(a, 10) returns t s.t. np.mean( a <= t ) = 0.1
        phat = np.bincount(digits[result_is_confident], minlength=10)
        phat = (phat + 0.) / np.sum(phat)
        threshold = np.percentile(phat, 5)
        ratio_not_covered = np.mean(phat <= threshold)
        C = 1. - ratio_not_covered


    return [C,C*np.mean(result_is_confident),np.mean(result_is_confident),C_actual]



"""
Used for computing support size as described in the 6.883 project report.

Based on:
Arora, Sanjeev, and Yi Zhang. "Do GANs actually learn the distribution? an empirical study." arXiv preprint arXiv:1706.08224 (2017).


For synthetic data sets:
   - will write a file containing the n closes points for each trial for each value of s
      n is min(s, 20)
   - will write a file listing the percentage of trial where a collision is found for each s
For mnist:
   - will save an image with the n stacked closest pairs for each trial for each value of s
      n is min(s, 20)

Inputs:
dataset: dataset name: {mnist, 2d_ring, 2d_grid, hd}
samples: a numpy array of the testing data. where len(samples) >= max(s_list)
s_list: the list of s values to test
work_dir: directory to save output
"""
def birthday_paradox(dataset, samples, s_list, work_dir):

  def _diff_signs(x1, x2):
    if x1 < 0 and x2 > 0:
      return True
    elif x1 > 0 and x2 < 0:
      return True
    return False

  if not os.path.isdir(work_dir):
    os.mkdir(work_dir)

  OUTPUT_FOLDER_NAME = os.path.join(work_dir, 'birthdayParadoxResults_'+dataset)

  if not os.path.isdir(OUTPUT_FOLDER_NAME):
    os.mkdir(OUTPUT_FOLDER_NAME)

  if 'mnist' in dataset:
    num_trials = 10
  else:
    num_trials = 50

  coll_per_s = [] 
  for s in s_list:
    trials_with_collision = 0
    for trial in range(num_trials):
        
        # get random samples
        tmp_replace=False
        if samples.shape[0]<s:
          tmp_replace=True
          
        ind=np.random.choice(np.arange(samples.shape[0]),size=s,replace=False)
        images=samples[ind]
        
        if 'mnist' in dataset:
          images=np.reshape(images,(images.shape[0], 28, 28))
          
        # remove noise frm hd dataset
        if 'hd' in dataset:
          images = images[:, :500]
    

        # get closest 20 pairs using euclidean distance
        topK=20 if s >= 20 else s
        queue = []
        n_image = images.shape[0]

        for i in range(n_image):
          for j in range(i+1, n_image):
            dist = np.sum((images[i] - images[j])**2)
            abs_dist = abs(dist)
            queue.append((abs_dist, copy.deepcopy(images[i]), copy.deepcopy(images[j])))
        queue = sorted(queue, key=lambda x: x[0])
        queue = queue[:topK]

        # find collisions
        if 'mnist' in dataset:
          final_image = []
          for idx in range(topK):
            dist, img1, img2 = queue[idx]
            joined_image = []
            for a in range(len(img1)):
              row = list(img1[a]) + list(img2[a])
              joined_image.append(row)
            final_image = final_image + joined_image
          final_image = np.array(final_image)
          file_name = 's_' + str(s) + '_trial_' + str(trial) + '.png'
          name = os.path.join(OUTPUT_FOLDER_NAME, file_name)
          scipy.misc.imsave(name, final_image)


        elif 'hd' in dataset:
          collision = False
          for idx in range(topK):
            dist, img1, img2 = queue[idx]
            if abs(dist) <= 2.25:
              collision = True
              break

          if collision:
            trials_with_collision += 1
              
          file_name = 's_' + str(s) + '_trial_' + str(trial) + '.npy'
          name = os.path.join(OUTPUT_FOLDER_NAME, file_name)
          top20_array = np.array([[i[1], i[2]] for i in queue])
          np.save(name, top20_array)

        else:
          collision = False
          for idx in range(topK):
            dist, img1, img2 = queue[idx]

            dims = len(img1)
            dims_match = 0
            for dim_img1, dim_img2 in zip(img1, img2):
              # different signs take abs of result
              if _diff_signs(dim_img1, dim_img2):
                diff = abs(dim_img1 - dim_img2)
              else:
                diff = abs(abs(dim_img1) - abs(dim_img2))
              if diff <= 0.05:
                dims_match += 1
                
            if dims_match == dims:
              collision = True
              break

          if collision:
            trials_with_collision += 1
  
          file_name = 's_' + str(s) + '_trial_' + str(trial) + '.npy'
          name = os.path.join(OUTPUT_FOLDER_NAME, file_name)
          top20_array = np.array([[i[1], i[2]] for i in queue])
          np.save(name, top20_array)
  
      # gather list of perc collision for each s for synthetic data
    if 'mnist' not in dataset:
      perc_collision = 100*(float(trials_with_collision)/num_trials)
      coll_per_s.append([s, perc_collision])

  if 'mnist' not in dataset:
    name = os.path.join(OUTPUT_FOLDER_NAME, 'percent_collisions.txt')
    file = open(name, 'w')
    file.write('s\tpercent_collision\n')
    for l in coll_per_s:
      to_write = str(l[0])+'\t'+str(l[1])+'\n'
      file.write(to_write)
    file.close()


