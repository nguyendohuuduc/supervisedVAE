import tensorflow as tf

# Gaussian MLP as conditional encoder
def classification_net(x,y,n_hidden,keep_prob):
    with tf.variable_scope("classification_net"):
        dim_y = int(y.get_shape()[1])
        input = tf.concat(axis=1,values = [x])
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(input, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)
        
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)
        
        w2 = tf.get_variable('w2', [h1.get_shape()[1], n_hidden], initializer=w_init)
        b2 = tf.get_variable('b2', [n_hidden], initializer=b_init)
        h2 = tf.matmul(h1, w2) + b2
        h2 = tf.nn.tanh(h2)
        h2 = tf.nn.dropout(h2, keep_prob)
        
        w3 = tf.get_variable('w3', [h2.get_shape()[1], n_hidden], initializer=w_init)
        b3 = tf.get_variable('b3', [n_hidden], initializer=b_init)
        h3 = tf.matmul(h2, w3) + b3
        h3 = tf.nn.tanh(h3)
        h3 = tf.nn.dropout(h3, keep_prob)
        
        w4 = tf.get_variable('w4', [h3.get_shape()[1], n_hidden], initializer=w_init)
        b4 = tf.get_variable('b4', [n_hidden], initializer=b_init)
        h4 = tf.matmul(h3, w4) + b4
        h4 = tf.nn.tanh(h4)
        h4 = tf.nn.dropout(h4, keep_prob)
        
        wo = tf.get_variable('wo', [h4.get_shape()[1], dim_y], initializer=w_init)
        bo = tf.get_variable('bo', [dim_y], initializer=b_init)
        ho = tf.matmul(h4, wo) + bo
        y_pred = tf.nn.softmax(ho)
    return y_pred

def gaussian_MLP_conditional_encoder(x, y, n_hidden, n_output, keep_prob):
    with tf.variable_scope("gaussian_MLP_encoder"):
        # concatenate condition and image
        dim_y = int(y.get_shape()[1])
        input = tf.concat(axis=1, values=[x, y])

        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        # w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden+dim_y], initializer=w_init)
        # b0 = tf.get_variable('b0', [n_hidden+dim_y], initializer=b_init)
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(input, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2], initializer=w_init)
        bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
        gaussian_params = tf.matmul(h1, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])

    return mean, stddev

# Bernoulli MLP as conditional decoder
def bernoulli_MLP_conditional_decoder(z, y, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
        # concatenate condition and latent vectors
        input = tf.concat(axis=1, values=[z, y])

        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(input, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return y

# Gateway
def autoencoder(x_hat, x, y, dim_img, dim_z, n_hidden, keep_prob):

    # encoding
    y_pred = classification_net(x_hat,y,n_hidden,keep_prob)

    mu, sigma = gaussian_MLP_conditional_encoder(x_hat, y_pred, n_hidden, dim_z, keep_prob)

    # sampling by re-parameterization technique
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    # decoding
    x_ = bernoulli_MLP_conditional_decoder(z, y_pred, n_hidden, dim_img, keep_prob)
    #x_ = bernoulli_MLP_conditional_decoder(z, y, n_hidden, dim_img, keep_prob)
    x_ = tf.clip_by_value(x_, 1e-8, 1 - 1e-8)

    # ELBO
    marginal_likelihood = tf.reduce_sum(x * tf.log(x_) + (1 - x) * tf.log(1 - x_), 1)
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)
    alpha = 25

    classification_error = tf.losses.softmax_cross_entropy(y,y_pred)

    ELBO = marginal_likelihood - KL_divergence

    # minimize loss instead of maximizing ELBO
    loss = -ELBO  + alpha * classification_error

    return x_, z, loss, -marginal_likelihood, KL_divergence,classification_error,y_pred

# Conditional Decoder (Generator)
def decoder(z, y, dim_img, n_hidden):

    x_ = bernoulli_MLP_conditional_decoder(z, y, n_hidden, dim_img, 1.0, reuse=True)

    return x_
    
def predict(x,y,n_hidden):
   y_pred = classification_net(x,y,n_hidden,1.0,reuse=True)
   return y_pred 