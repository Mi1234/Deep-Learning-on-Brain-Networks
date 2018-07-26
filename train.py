import sys, os
# sys.path.insert(0, '..')
sys.path.insert(0,'/home/miranda/Code/cnn_graph/lib')
# import models_cnn as models
import models
import graph, coarsening, utils

import tensorflow as tf
import numpy as np
import time
import timeit
import gzip, pickle

flags = tf.app.flags
FLAGS = flags.FLAGS

# Graphs.
flags.DEFINE_integer('number_edges', 8, 'Graph: minimum number of edges per vertex.')
flags.DEFINE_string('metric', 'euclidean', 'Graph: similarity measure (between features).')
# TODO: change cgcnn for combinatorial Laplacians.
flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
flags.DEFINE_integer('coarsening_levels', 4, 'Number of coarsened graphs.')

# Directories.
flags.DEFINE_string('dir_data', os.path.join('..', 'data', 'connectome'), 'Directory to store data.')


# # Feature graph


def grid_graph(m, corners=False):
    z = graph.grid(m)
    dist, idx = graph.distance_sklearn_metrics(z, k=FLAGS.number_edges, metric=FLAGS.metric)
    A = graph.adjacency(dist, idx)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz//2, FLAGS.number_edges*m**2//2))
    return A

t_start = timeit.default_timer()
A = grid_graph(58, corners=False)
A = graph.replace_random_edges(A, 0)
graphs, perm = coarsening.coarsen(A, levels=FLAGS.coarsening_levels, self_connections=False)
L = [graph.laplacian(A, normalized=True) for A in graphs]
print('Execution time: {:.2f}s'.format(timeit.default_timer() - t_start))
graph.plot_spectrum(L)
del A


# # Data


with gzip.open('/home/miranda/Code/aae_deep_decision_neural_network/data/syn_adhd_train_data_quater_scale.pkl.gz', 'rb') as f:  
    X_train = pickle.load(f)
X = zip(*X_train)[0]
labels = zip(*X_train)[1]
X = np.asarray(X)
y = np.asarray(labels)

# y = []
# for i in labels:
#     if i == 0:
#         y.append([0,1])
#     else:
#         y.append([1,0])

# y = np.asarray(y)

train_data = X[0:5000, :]
train_labels = y[0:5000]
val_data = X[5000:6000, :]
val_labels = y[5000:6000]
test_data = X[6000:6732, :]
test_labels = y[6000:6732]


t_start = timeit.default_timer()
train_data = coarsening.perm_data(train_data, perm)
val_data = coarsening.perm_data(val_data, perm)
test_data = coarsening.perm_data(test_data, perm)
print('Execution time: {:.2f}s'.format(timeit.default_timer() - t_start))
del perm


# # Neural networks

if False:
    K = 5  # 5 or 5^2
    t_start = timeit.default_timer()
    mnist.test._images = graph.lanczos(L, mnist.test._images.T, K).T
    mnist.train._images = graph.lanczos(L, mnist.train._images.T, K).T
    model = lgcnn2_1(L, F=10, K=K)
    print('Execution time: {:.2f}s'.format(timeit.default_timer()- t_start))
    ph_data = tf.placeholder(tf.float32, (FLAGS.batch_size, mnist.train.images.shape[1], K), 'data')



common = {}
common['dir_name']       = 'mnist/'
common['num_epochs']     = 20
common['batch_size']     = 32
common['decay_steps']    = 5000 / common['batch_size']
common['eval_frequency'] = 30 * common['num_epochs']
common['brelu']          = 'b1relu'
common['pool']           = 'mpool1'
C = 2  # number of classes

model_perf = utils.model_perf()


# Common hyper-parameters for networks with one convolutional layer.
# common['dropout']        = 0.7
# common['learning_rate']  = 0.02
# common['decay_rate']     = 0.95
# common['momentum']       = 0.9
# common['F']              = [20]
# common['K']              = [20]
# common['p']              = [1]
# common['M']              = [C]




# With 'chebyshev2' and 'b2relu', it corresponds to cgcnn2_2(L[0], F=10, K=20).
# if True:
#     name = 'cgconv_softmax'
#     params = common.copy()
#     params['dir_name'] += name
#     params['filter'] = 'chebyshev5'
# #    params['filter'] = 'chebyshev2'
#     params['brelu'] = 'b2relu'
#     model_perf.test(models.cgcnn(L, **params), name, params,
#                     train_data, train_labels, val_data, val_labels, test_data, test_labels)




# Common hyper-parameters for 2 layer convolutional neural networks.
common['regularization'] = 5e-4
common['dropout']        = 0.5
common['learning_rate']  = 1e-2  # 0.03 in the paper but sgconv_sgconv_fc_softmax has difficulty to converge
common['decay_rate']     = 0.95
common['momentum']       = 0.7
common['F']              = [60, 100]
common['K']              = [25, 25]
common['p']              = [4, 4]
common['M']              = [625, C]
common['N_TREE']         = 5
common['DEPTH']          = 3                 
common['N_LEAF']         = 2 ** (common['DEPTH'] + 1)
common['p_keep_hidden']  = 0.5


if True:
    name = 'cgconv_cgconv_fc_softmax'  # 'Chebyshev'
    params = common.copy()
    params['dir_name'] += name
    params['filter'] = 'chebyshev5'
    model_perf.test(models.cgcnn(L, **params), name, params,
                    train_data, train_labels, val_data, val_labels, test_data, test_labels)


model_perf.show()


if False:
    grid_params = {}
    data = (train_data, train_labels, val_data, val_labels, test_data, test_labels)
    utils.grid_search(params, grid_params, *data, model=lambda x: models.cgcnn(L,**x))

