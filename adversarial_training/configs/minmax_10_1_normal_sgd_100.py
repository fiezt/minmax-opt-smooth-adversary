# path to dataset
data_dir = './dataset'

# path to results
folder = './results/minmax_10_1_normal_sgd_100'

# training algorithm: concave, pgd, or minmax
alg = 'normal'

# number of training epochs
num_epochs = 100

# size of each batch of mnist images
batch_size = 50

# number of epochs to wait before attacking
loss_every = 1

# number of epochs to wait before attacking
attack_every = 5

# number of epochs to wait before saving model
save_every = 5

# learning_rate for outer optimization
learning_rate_outer = 1e-4

# learning_rate for inner optimization
learning_rate_inner = 4

# number of inner optimization steps
num_inner_steps = 10

# epsilon parameter used by algorithm
eps = 0.3