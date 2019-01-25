import tensorflow as tf
import numpy as np
import argparse

"""parsing and configuration"""
def parse_args():
    desc = "nn"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP')

    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')


    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


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

def read_from_csv():
    train = np.loadtxt(open("500-dim-z-one-hot-labelmnist-train.csv", "rb"), delimiter=",", skiprows=0)
    trainX = train[:,:-10]
    trainYone_hot = train[:,-10:].astype(np.int32)
    test = np.loadtxt(open("500-dim-z-one-hot-labelmnist-test.csv", "rb"), delimiter=",", skiprows=0)
    testX = test[:,:-10]
    testYone_hot = test[:,-10:].astype(np.int32)
    return np.concatenate([trainX,trainYone_hot],axis=1),testX,testYone_hot

    
def nn_model(x_hat, y, n_hidden, keep_prob):
    y_pred = classification_net(x_hat,y,n_hidden,keep_prob)
    classification_error = tf.losses.softmax_cross_entropy(y,y_pred)
    return y_pred,classification_error
    
def main(args):
    n_hidden = args.n_hidden
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate
    
    
    train_total_data, test_data, test_labels = read_from_csv()
    n_train = train_total_data.shape[0]
    n_samples = n_train
    n_test = test_data.shape[0]
    NUM_LABELS = 10
    n_features = train_total_data.shape[1] - NUM_LABELS
    

    
    x_hat = tf.placeholder(tf.float32, shape=[None, n_features], name='input_img')
    y = tf.placeholder(tf.float32, shape=[None, NUM_LABELS], name='target_labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    logit,loss = nn_model(x_hat,y,n_hidden,keep_prob)
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    total_batch = int(n_train / batch_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})
        for epoch in range(n_epochs):
            np.random.shuffle(train_total_data)
            train_data_ = train_total_data[:, :-NUM_LABELS]
            train_labels_ = train_total_data[:, -NUM_LABELS:]
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_data_[offset:(offset + batch_size), :]
                batch_ys_input = train_labels_[offset:(offset + batch_size)]
                batch_xs_target = batch_xs_input
                
                _,_,classification_loss= sess.run(
                    (train_op,logit, loss),
                    feed_dict={x_hat: batch_xs_input, y: batch_ys_input, keep_prob : 0.9})
                    
            print("epoch %d: class_loss %03.2f " % (epoch, classification_loss))
            if epoch == n_epochs-1:
                test_iters = int(n_test/batch_size)
                wrong_total = 0
                for curr_iter in range(test_iters):
                    offset = (curr_iter * batch_size) % (n_test)
                    batch_xs_input = test_data[offset:(offset + batch_size), :]
                    batch_ys_input = test_labels[offset:(offset + batch_size),:]
                    batch_xs_target = batch_xs_input
                    logit_result =sess.run(logit,feed_dict={x_hat:batch_xs_input,y:batch_ys_input,keep_prob:1.0})
                    prediction = tf.argmax(logit_result,axis=1)
                    true_label = tf.argmax(batch_ys_input,axis=1)
                    diff = prediction-true_label
                    wrong = tf.count_nonzero(diff).eval()
                    wrong_total += wrong
                print("classification accuracy ",(n_test-wrong_total)/float(n_test))

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)    