'''
Author: Onkar Jadhav
Python script for a PINNs of BE
'''

# Import TensorFlow and NumPy
import tensorflow as tf
import numpy as np
# import keras.backend as K


# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

# Set constants
pi = tf.constant(np.pi, dtype=DTYPE)
viscosity = 0 # 0.00001 #.

# Define initial condition
def fun_u_0(x):
    return 1 if tf.math.greater(tf.math.abs(1/3), tf.math.abs(x)) else 0 #-tf.sin(pi * x) #

# Define boundary condition
# def fun_u_b(t, x):
#     n = x.shape[0]
#     return tf.zeros((n,1), dtype=DTYPE)

# Define residual of the PDE
def fun_r(t, x, u, u_t, u_x, u_xx):
    # LAM = 1/(((tf.math.abs(u_x) - u_x) +1))
    return u_t + u_x # u_t + u * u_x - viscosity * u_xx

# Set number of data points
N_0 = 100
N_b = 100
N_r = 10000 #10000

# Set boundary
tmin = 0.
tmax = 10.
xmin = -1.
xmax = 1.
#%%
# Lower bounds
lb = tf.constant([tmin, xmin], dtype=DTYPE)
# Upper bounds
ub = tf.constant([tmax, xmax], dtype=DTYPE)

# Set random seed for reproducible results
tf.random.set_seed(0)

# # Draw uniform sample points for initial boundary data
# t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[0]
x_0 = tf.random.uniform((N_0,1), lb[1], ub[1], dtype=DTYPE)
# X_0 = tf.concat([t_0, x_0], axis=1)

xx_0 = x_0.numpy()
u_0 = []
# Evaluate intitial condition at x_0
for i in range(len(xx_0)):    
    u_00 = fun_u_0(xx_0[i])
    u_0 = np.append(u_0, u_00)
    

# Draw uniform sample points for initial boundary data
t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[0]
# x_0 = tf.random.uniform((N_0,1), lb[1], ub[1], dtype=DTYPE)
X_0 = tf.concat([t_0, x_0], axis=1)

# Evaluate intitial condition at x_0
# u_0 = fun_u_0(x_0)
    
u_0 = tf.convert_to_tensor(u_0, dtype=DTYPE)

# Boundary data
t_b = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE)
# x_b = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=DTYPE)

K = int(N_b/2)

x_b = np.array([xmin]*K + [xmax]*K)
np.random.shuffle(x_b)
x_b = x_b.reshape(-1,1)
x_b = tf.convert_to_tensor(x_b, dtype=DTYPE)
X_b = tf.concat([t_b, x_b], axis=1)

XX_b = X_b.numpy()
X_b1 = XX_b[XX_b[:,1] == -1, :]
X_b2 = XX_b[XX_b[:,1] == 1, :]

X_b1 = tf.convert_to_tensor(X_b1, dtype=DTYPE)
X_b2 = tf.convert_to_tensor(X_b2, dtype=DTYPE)



# Evaluate boundary condition at (t_b,x_b)
# u_b = fun_u_b(t_b, x_b)

# Draw uniformly sampled collocation points
t_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
x_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
X_r = tf.concat([t_r, x_r], axis=1)

# X_r = tf.concat([X_r], axis=0) #, X_b1, X_b2

# Collect boundary and initial data in lists
X_data = [X_0, X_b]
X_b_numpy = X_b.numpy()
# u_data = [u_0, u_b]
u_data = [u_0]

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9,6))
plt.scatter(t_0, x_0, c=u_0, marker='X', vmin=-1, vmax=1)
# plt.scatter(t_b, x_b, marker='X', vmin=-1, vmax=1)
plt.scatter(t_r, x_r, c='r', marker='.', alpha=0.1)
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.title('Positions of collocation points and boundary data');
#plt.savefig('Xdata_Burgers.pdf', bbox_inches='tight', dpi=300)
#%%
def init_model(num_hidden_layers=5, num_neurons_per_layer=20):
    # Initialize a feedforward neural network
    model = tf.keras.Sequential()

    # Input is two-dimensional (time + one spatial dimension)
    model.add(tf.keras.Input(2))

    # Introduce a scaling layer to map input to [lb, ub]
    scaling_layer = tf.keras.layers.Lambda(
                lambda x: 1.0*(x - lb)/(ub - lb))
    model.add(scaling_layer)

    # Append hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get('sigmoid'),
            kernel_initializer='glorot_normal'))

    # Output is one-dimensional
    model.add(tf.keras.layers.Dense(1))
    
    return model

def get_r(model, X_r):
    
    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        # Split t and x to compute partial derivatives
        t, x = X_r[:, 0:1], X_r[:,1:2]

        # Variables t and x are watched during tape
        # to compute derivatives u_t and u_x
        tape.watch(t)
        tape.watch(x)

        # Determine residual 
        u = model(tf.stack([t[:,0], x[:,0]], axis=1))

        # Compute gradient u_x within the GradientTape
        # since we need second derivatives
        u_x = tape.gradient(u, x)
        # u2 = u*u
        # u2_x = tape.gradient(u2, x)
            
        u_t = tape.gradient(u, t)
    u_xx = tape.gradient(u_x, x)

    del tape

    return fun_r(t, x, u, u_t, u_x, u_xx)

def get_ux(model, X_r):
    
    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        # Split t and x to compute partial derivatives
        t, x = X_r[:, 0:1], X_r[:,1:2]

        # Variables t and x are watched during tape
        # to compute derivatives u_t and u_x
        tape.watch(t)
        tape.watch(x)

        # Determine residual 
        u = model(tf.stack([t[:,0], x[:,0]], axis=1))

        # Compute gradient u_x within the GradientTape
        # since we need second derivatives
        u_x = tape.gradient(u, x)

    del tape

    return u_x, u


def compute_loss(model, X_r, X_data, u_data):
    
    # Compute phi^r
    r = get_r(model, X_r)
    phi_r = tf.reduce_mean(tf.square(r))
    
    # Initialize loss
    loss = phi_r
    
    # Add phi^0 and phi^b to the loss
    # for i in range(len(X_data)):
    #     u_pred = model(X_data[i])
    #     loss += tf.reduce_mean(tf.square(u_data[i] - u_pred))
    
    u_pred = model(X_data[0])
    loss += tf.reduce_mean(tf.square(u_data[0] - u_pred))
  
    # X_data1_numpy = X_data[1][:,1]
    
    # try:
    #     X_data1_numpy = X_data1_numpy.numpy()
    # except:
    #     print(X_data1_numpy)
    #     X_data1_numpy = X_data1_numpy.numpy()
    
    X_data1_numpy = X_b_numpy[:,1]
    for i in range(len(X_data1_numpy)):
        if X_data1_numpy[i] == -1:
            u_BC1 = model(X_data[1])
            print(u_BC1)
        else:
            u_BC2 = model(X_data[1])
            print(u_BC2)

    # u_BC1 = r[N_r:N_r+K,:]
    # u_BC2 = r[N_r+K:N_r+K+K,:]
    loss += tf.reduce_mean(tf.square(u_BC1 - u_BC2))
           
    return loss

# def getBCres(model, X_r, X_data, u_data):
#     u_BC1 = model(X_data[1])
#     u_BC2 = model(X_data[2])
#     u_pred = model(X_data[0])
#     return u_BC1, u_BC2, u_pred

def get_grad(model, X_r, X_data, u_data):
    
    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables
        tape.watch(model.trainable_variables)
        loss = compute_loss(model, X_r, X_data, u_data)

    g = tape.gradient(loss, model.trainable_variables)
    del tape

    return loss, g

# Initialize model aka u_\theta
model = init_model()

# We choose a piecewise decay of the learning rate, i.e., the
# step size in the gradient descent type algorithm
# the first 1000 steps use a learning rate of 0.01
# from 1000 - 3000: learning rate = 0.001
# from 3000 onwards: learning rate = 0.0005

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])

# Choose the optimizer
optim = tf.keras.optimizers.Adam(learning_rate=lr)

from time import time

# Define one training step as a TensorFlow function to increase speed of training
@tf.function
def train_step():
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = get_grad(model, X_r, X_data, u_data)
    
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, model.trainable_variables))
    
    return loss

# Number of training epochs
N = 10000
hist = []

# Start timer
t0 = time()

for i in range(N+1):
    
    loss = train_step()
    
    # Append current loss to hist
    hist.append(loss.numpy())
    
    # Output current loss after 50 iterates
    if i%50 == 0:
        print('It {:05d}: loss = {:10.8e}'.format(i,loss))

# Print Ux
UU, Ux = get_ux(model, X_r)   

# u_BC1, u_BC2, upred = getBCres(model, X_r, X_data, u_data)  
r = get_r(model, X_r)

# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))

from mpl_toolkits.mplot3d import Axes3D

# Set up meshgrid
N = 600
tspace = np.linspace(lb[0], ub[0], 100)
xspace = np.linspace(lb[1], ub[1], N + 1)
T, X = np.meshgrid(tspace, xspace)
Xgrid = np.vstack([T.flatten(),X.flatten()]).T

# Determine predictions of u(t, x)
upred = model(tf.cast(Xgrid,DTYPE))

# Reshape upred
U = upred.numpy().reshape(N+1,100)

# Surface plot of solution u(t,x)
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111) # projection='3d'
#ax.plot_surface(T[:,0], X[:,0], U[:,0], cmap='viridis');
ax.plot(X[:,99], U[:,99])
#ax.view_init(35,35)
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
#ax.set_zlabel('$u_\\theta(t,x)$')
ax.set_title('Solution of Burgers equation');
#plt.savefig('Burgers_Solution.pdf', bbox_inches='tight', dpi=300);

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
ax.semilogy(range(len(hist)), hist,'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi_{n_{epoch}}$');
