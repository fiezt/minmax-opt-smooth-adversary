z_dim = 16
g_inp = z_dim
g_hid = 32
g_out = 2

d_inp = g_out
d_hid = 32
d_out = 1

minibatch_size = 512
optim_betas = (0.5, 0.999)
num_iterations = 150000
log_interval = 2000
d_steps = 1
g_steps = 1
seed = 123456
use_higher = True
restart = True
unrolled_steps = 15
objective = 'sgan'
optimizer = 'sgd'
d_learning_rate = 0.5
g_learning_rate = 0.5
prefix = 'sim3_sgan'