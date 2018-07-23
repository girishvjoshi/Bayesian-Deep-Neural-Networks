'''
Author: Girish Joshi
Date: 07/23/2018
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# BNN Hyperparameters
sigma_prior = 0.1
epsilon_prior = 0.1
lr = 0.001
n_epochs = 4000
stddev_var = 0.001

    
class VariationalDense:
    
    def __init__(self, n_in, n_out):
        
        self.W_mu = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=stddev_var))
        self.W_logsigma = tf.Variable(tf.truncated_normal([n_in, n_out], mean = 0., stddev=stddev_var))
        self.b_mu = tf.Variable(tf.zeros([n_out]))
        self.b_logsigma = tf.Variable(tf.zeros([n_out]))
        
        self.epsilon_w = self.get_random([n_in,n_out],mu=0., std_dev=epsilon_prior)
        self.epsilon_b = self.get_random([n_out], mu=0., std_dev =epsilon_prior)
        
        self.W = self.W_mu + tf.multiply(tf.log(1. + tf.exp(self.W_logsigma)), self.epsilon_w)
        self.b = self.b_mu + tf.multiply(tf.log(1. + tf.exp(self.b_logsigma)), self.epsilon_b)
        
    def __call__(self, x, activation=tf.identity):
        output = activation(tf.matmul(x,self.W) + self.b)
        output = tf.squeeze(output)
        return output
    
    
    def log_gaussian(self, x, mu, sigma):
        return -0.5*tf.log(2*np.pi)-tf.log(sigma)-(x-mu)**2/(2*sigma**2)

    def get_random(self, shape, mu, std_dev):
        return tf.random_normal(shape, mean=mu, stddev=std_dev)
    
    def regularization(self):
        
        sample_log_pw, sample_log_qw= 0. , 0.
        
        sample_log_pw += tf.reduce_sum(self.log_gaussian(self.W, 0., sigma_prior))
        sample_log_pw += tf.reduce_sum(self.log_gaussian(self.b, 0., sigma_prior))
        
        sample_log_qw += tf.reduce_sum(self.log_gaussian(self.W, self.W_mu, tf.log(1. + tf.exp(self.W_logsigma))))
        sample_log_qw += tf.reduce_sum(self.log_gaussian(self.b, self.b_mu, tf.log(1. + tf.exp(self.b_logsigma))))
        
        regulizer = tf.reduce_sum((sample_log_qw-sample_log_pw))
        
        return regulizer

# Create the Model
n_sample = 20
X = np.random.normal(size=(n_sample,1))
y = np.random.normal(np.cos(5.*X) / (np.abs(X) + 1.), 0.1).ravel()
X_pred = np.atleast_2d(np.linspace(-3.,3.,num=100)).T
X = np.hstack((X, X**2, X**3))
X_pred = np.hstack((X_pred, X_pred**2, X_pred**3)) 

feature_size = X.shape[1]
n_hidden = 100

# Place holder of Network inputs and outputs
model_x = tf.placeholder(tf.float32, shape=[None, feature_size])
model_y = tf.placeholder(tf.float32, shape=[None])

#Define neural network
net_Layer_input = VariationalDense(feature_size, n_hidden)
net_Layer_hidden = VariationalDense(n_hidden, n_hidden)
net_Layer_output = VariationalDense(n_hidden, 1)

Input_Layer_output = net_Layer_input(model_x, tf.nn.relu)
Hidden_Layer_output = net_Layer_hidden(Input_Layer_output, tf.nn.relu)
net_pred =  net_Layer_output(Hidden_Layer_output)

# Define ELBO
sample_log_Likelihood = tf.reduce_sum(net_Layer_output.log_gaussian(model_y, net_pred, sigma_prior))

regularization_term = net_Layer_input.regularization() + net_Layer_output.regularization() + net_Layer_output.regularization()

elbo = -sample_log_Likelihood + regularization_term / n_hidden

train_step = tf.train.AdamOptimizer(lr).minimize(elbo)

# Mean Square Error (Network Performance)
model_mse = tf.reduce_sum(tf.square(model_y - net_pred))/n_sample

# Train the MOdel
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(n_epochs):
        sess.run(train_step, feed_dict={model_x: X, model_y: y})
        if i % 100 == 0:
            mse = sess.run(model_mse, feed_dict={model_x: X, model_y: y})
            print('iteration {}: Mean Squared Error: {:.4f}'.format(i,mse))

    n_post = 1000
    y_post = np.zeros((n_post, X_pred.shape[0]))
    for i in range(n_post):
        y_post[i] = sess.run(net_pred, feed_dict={model_x: X_pred})


#plot the Results
plt.figure(figsize=(8,6))
for i in range(n_post):
    plt.plot(X_pred[:,0], y_post[i], 'b-', alpha = 1./200 )
plt.plot(X[:,0],y,'r.')
plt.grid()
plt.show()

