'''
Author: Onkar Jadhav
PINNs for one-factor Hull-White model, covection diffusion reaction equation. 
'''

# Import TensorFlow and NumPy
import tensorflow as tf
import numpy as np
import math
# import keras.backend as K

# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

# Set constants
pi = tf.constant(np.pi, dtype=DTYPE)
viscosity = 0.01/pi #0.0001/pi # 0.00001 #.

# Define initial condition
def fun_u_0(x):
    n = x.shape[0]
    return tf.ones((n,1), dtype=DTYPE) #tf.sin(pi * x)  #10*x - 5*(x**2)

# Define boundary condition
def fun_u_b(t, x):
    n = x.shape[0]
    return tf.ones((n,1), dtype=DTYPE)

sigma = 0.006*0.006*0.5
theta = 0.005
a = 0.1

def fun_r(t, x, u, u_t, u_x, u_xx):
    return u_t + (theta - a*x)*u_x + sigma*u_xx - x*u

# Set number of data points
N_0 = 100
N_b = 100
N_r = 10000

# Set boundary
tmin = 0.
tmax = 10.
xmin = -0.1
xmax = 0.1

# Lower bounds
lb = tf.constant([tmin, xmin], dtype=DTYPE)
# Upper bounds
ub = tf.constant([tmax, xmax], dtype=DTYPE)

# Set random seed for reproducible results
tf.random.set_seed(0)

# Draw uniform sample points for initial boundary data
t_0 = tf.ones((N_0,1), dtype=DTYPE)*ub[0]
x_0 = tf.random.uniform((N_0,1), lb[1], ub[1], dtype=DTYPE)
X_0 = tf.concat([t_0, x_0], axis=1)

# Evaluate intitial condition at x_0
u_0 = fun_u_0(x_0)

# Boundary data
t_b = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE)
x_b = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=DTYPE)
# x_b = lb[1] * tf.keras.backend.random_bernoulli((N_b,1), 1, dtype=DTYPE)
X_b = tf.concat([t_b, x_b], axis=1)

# Draw uniformly sampled collocation points
t_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
x_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
X_r = tf.concat([t_r, x_r], axis=1)

MT = [1,2,3,4,5,6,7,8,9,10]
#%%
# Collect boundary and initial data in lists
X_data = [X_0, X_b]
# u_data = [u_0, u_b]
u_data = [u_0]

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9,6))
plt.scatter(t_0, x_0, c=u_0, marker='X', vmin=0, vmax=1)
# plt.scatter(t_b, x_b, c=u_b, marker='X', vmin=-1, vmax=1)
plt.scatter(t_r, x_r, c='r', marker='.', alpha=0.1)
plt.xlabel('$t$')
plt.ylabel('$x$')

plt.title('Positions of collocation points and boundary data');
#plt.savefig('Xdata_Burgers.pdf', bbox_inches='tight', dpi=300)

def init_model(num_hidden_layers=5, num_neurons_per_layer=20):
    # Initialize a feedforward neural network
    model = tf.keras.Sequential()

    # Input is two-dimensional (time + one spatial dimension)
    model.add(tf.keras.Input(2))

    # Introduce a scaling layer to map input to [lb, ub]
    scaling_layer = tf.keras.layers.Lambda(
                lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
    model.add(scaling_layer)

    # Append hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get('tanh'),
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
        
        u = model(tf.stack([t[:,0], x[:,0]], axis=1))
        
        # u = u + 0.025
        
        # for i in range(len(TempTr)):
        #     if np.any(np.isclose(TempTr[i], MT, rtol=5e-05)):
        #         print(TempTr[i])
        #         u = tf.add(u,0.025)
                # u = tf.math.maximum(u, tf.ones((u.shape[0],1),dtype=DTYPE))
        
        # TempTTr = TTr % 360
        
        
        # if math.isclose(any(TTr), 0.1):
        #     # m = np.repeat(m,len(X.flatten()))
        #     Xgrid = np.vstack([TTr,XXr]).T
        #     u = model(tf.cast(Xgrid, DTYPE))
        #     u = tf.math.maximum(u, tf.ones((u.shape[0],1),dtype=DTYPE))
        #     u = u + 0.025
        # else:
        #     # m = np.repeat(m,len(X.flatten()))
        #     Xgrid = np.vstack([TTr,XXr]).T
        #     u = model(tf.cast(Xgrid, DTYPE))
        

        # T_r = t_r.numpy()
        # # Determine residual 
        
        # if math.isclose((T_r % 360), 0.001):
        #     u = model(tf.stack([t[:,0], x[:,0]], axis=1))
        #     u = tf.math.maximum(u, tf.ones((u.shape[0],1),dtype=DTYPE))
        # else:
        #     u = model(tf.stack([t[:,0], x[:,0]], axis=1))

        # Compute gradient u_x within the GradientTape
        # since we need second derivatives
        u_x = tape.gradient(u, x)
            
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

def get_uxb(model, X_b):
    
    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        # Split t and x to compute partial derivatives
        tb, xb = X_b[:, 0:1], X_b[:,1:2]

        # Variables t and x are watched during tape
        # to compute derivatives u_t and u_x
        tape.watch(tb)
        tape.watch(xb)

        # Determine residual 
        ub = model(tf.stack([tb[:,0], xb[:,0]], axis=1))

        # Compute gradient u_x within the GradientTape
        # since we need second derivatives
        u_xb = tape.gradient(ub, xb)
        
    # u_xxb = tape.gradient(u_xb, xb)

    del tape

    return u_xb


def compute_loss(model, X_r, X_data, u_data):
    
    # Compute phi^r
    r = get_r(model, X_r)
    phi_r = tf.reduce_mean(tf.square(r))
    
    # Initialize loss
    loss_m = phi_r
    
    u_pred = model(X_data[0])
    loss_i = tf.reduce_mean(tf.square(u_data[0] - u_pred))
    
    u_xb = get_uxb(model, X_b)
    loss_b = tf.reduce_mean(tf.square(u_xb-0))
    # loss_b = u_xb
    
    loss = loss_m + loss_i + loss_b
    
    return loss

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
UUx, UU = get_ux(model, X_r)        

# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))

from mpl_toolkits.mplot3d import Axes3D

#%%
# Set up meshgrid
N = 300
tspace = np.linspace(lb[0], ub[0], N + 1)
xspace = np.linspace(lb[1], ub[1], N + 1)
# # tspace = t_r.numpy()
# # xspace = x_r.numpy()
T, X = np.meshgrid(tspace, xspace)
Xgrid = np.vstack([T.flatten(),X.flatten()]).T


upred = model(tf.cast(Xgrid,DTYPE))

# # Reshape upred
U = upred.numpy().reshape(N+1,N+1)

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, X, U, cmap='viridis');
ax.view_init(35,35)
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_zlabel('$u_\\theta(t,x)$')
ax.set_title('Solution of Burgers equation');

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111) # projection='3d'
#ax.plot_surface(T[:,0], X[:,0], U[:,0], cmap='viridis');
ax.plot(T[300,:], U[300, :])
# ax.invert_yaxis()
#ax.view_init(35,35)
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
#ax.set_zlabel('$u_\\theta(t,x)$')
ax.set_title('Solution of Burgers equation');

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
ax.semilogy(range(len(hist)), hist,'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi_{n_{epoch}}$');

#%%