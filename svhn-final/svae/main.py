import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from glob import glob
from keras.datasets import cifar10
from keras.models import Model,Sequential
from keras.layers import Dense, Input, Lambda, Reshape, Dropout, Flatten, Activation, Concatenate
from keras.optimizers import SGD, Adam, RMSprop
from keras import losses
import keras.backend as K
from keras import initializers
from keras.layers.normalization import BatchNormalization
from keras import metrics
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D 
from keras import regularizers
import scipy.io
from scipy.misc import imsave

NUM_CLASS = 10

np.random.seed(0)
train_data = scipy.io.loadmat('train_32x32.mat')
test_data = scipy.io.loadmat('test_32x32.mat')
X_train = train_data['X']/255.0
y_train = train_data['y']
X_test = test_data['X']/255.0
y_test = test_data['y']
X_train = np.transpose(X_train,(3,0,1,2))
X_test = np.transpose(X_test,(3,0,1,2))

temp = np.zeros((y_train.shape[0],NUM_CLASS))
for i in range(y_train.shape[0]):
    temp[i,y_train[i]%10] = 1
y_train = temp


temp = np.zeros((y_test.shape[0],NUM_CLASS))
for i in range(y_test.shape[0]):
    temp[i,y_test[i]%10] = 1
y_test = temp

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


def vae_loss(X_true,X_pred):
    recontruction_loss = K.mean(K.binary_crossentropy(X_true, X_pred))
    latent_loss = -0.5 * K.mean(1 + z_std_sq_log - K.square(z_mean) - K.exp(z_std_sq_log), axis=-1 )
    return recontruction_loss + 0.01*latent_loss  
    
def classification_loss(y_true,y_pred):
    return K.categorical_crossentropy(y_true,y_pred)
    
def recon_error(X_true,X_pred):
    return K.sum(K.square(X_true - X_pred)) / K.sum(K.square(X_true))
      
    
    
# Dimension of Latent Representation
dim_representation = 500

b_f = 128
# ENCODER
input_vae = Input(shape=(32,32,3))

# encoder_hidden1 = Conv2D(filters = b_f, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(input_vae)
# encoder_hidden1 = BatchNormalization()(encoder_hidden1)
# encoder_hidden1 = Activation('relu')(encoder_hidden1)

# encoder_hidden2 = Conv2D(filters = b_f*2, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(encoder_hidden1)
# encoder_hidden2 = BatchNormalization()(encoder_hidden2)
# encoder_hidden2 = Activation('relu')(encoder_hidden2)

# encoder_hidden3 = Conv2D(filters = b_f*4, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(encoder_hidden2)
# encoder_hidden3 = BatchNormalization()(encoder_hidden3)
# encoder_hidden3 = Activation('relu')(encoder_hidden3)

encoder_hidden1 = Conv2D(filters = b_f, kernel_size = 2, strides = (1,1), padding = 'same', kernel_initializer = 'he_normal' )(input_vae)
encoder_hidden1 = MaxPooling2D(pool_size = 2)(encoder_hidden1)
encoder_hidden1 = BatchNormalization()(encoder_hidden1)
encoder_hidden1 = Activation('relu')(encoder_hidden1)

encoder_hidden2 = Conv2D(filters = b_f*2, kernel_size = 2, strides = (1,1), padding = 'same', kernel_initializer = 'he_normal' )(encoder_hidden1)
encoder_hidden2 = MaxPooling2D(pool_size = 2)(encoder_hidden2)
encoder_hidden2 = BatchNormalization()(encoder_hidden2)
encoder_hidden2 = Activation('relu')(encoder_hidden2)

encoder_hidden3 = Conv2D(filters = b_f*4, kernel_size = 2, strides = (1,1), padding = 'same', kernel_initializer = 'he_normal' )(encoder_hidden2)
encoder_hidden3 = MaxPooling2D(pool_size = 2)(encoder_hidden3)
encoder_hidden3 = BatchNormalization()(encoder_hidden3)
encoder_hidden3 = Activation('relu')(encoder_hidden3)


encoder_hidden4 = Flatten()(encoder_hidden3)


# classification net

input_CNN = Input(shape=(32,32,3))

# CNN_hidden1 = Conv2D(filters = b_f, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(input_CNN)
# CNN_hidden1 = BatchNormalization()(CNN_hidden1)
# CNN_hidden1 = Activation('relu')(CNN_hidden1)

# CNN_hidden2 = Conv2D(filters = b_f*2, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal')(CNN_hidden1)
# CNN_hidden2 = BatchNormalization()(CNN_hidden2)
# CNN_hidden2 = Activation('relu')(CNN_hidden2)


# CNN_hidden3 = Conv2D(filters = b_f*4, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal')(CNN_hidden2)
# CNN_hidden3 = BatchNormalization()(CNN_hidden3)
# CNN_hidden3 = Activation('relu')(CNN_hidden3)



CNN_hidden1 = Conv2D(filters = b_f, kernel_size = 2, strides = (1,1), padding = 'same', kernel_initializer = 'he_normal' )(input_CNN)
CNN_hidden1 = MaxPooling2D(pool_size = 2)(CNN_hidden1)
CNN_hidden1 = BatchNormalization()(CNN_hidden1)
CNN_hidden1 = Activation('relu')(CNN_hidden1)

CNN_hidden2 = Conv2D(filters = b_f*2, kernel_size = 2, strides = (1,1), padding = 'same', kernel_initializer = 'he_normal')(CNN_hidden1)
CNN_hidden2 = MaxPooling2D(pool_size = 2)(CNN_hidden2)
CNN_hidden2 = BatchNormalization()(CNN_hidden2)
CNN_hidden2 = Activation('relu')(CNN_hidden2)


CNN_hidden3 = Conv2D(filters = b_f*4, kernel_size = 2, strides = (1,1), padding = 'same', kernel_initializer = 'he_normal')(CNN_hidden2)
CNN_hidden3 = MaxPooling2D(pool_size = 2)(CNN_hidden3)
CNN_hidden3 = BatchNormalization()(CNN_hidden3)
CNN_hidden3 = Activation('relu')(CNN_hidden3)

CNN_hidden4 = Flatten()(CNN_hidden3)

CNN_hidden5 = Dense(b_f*4, activation= 'relu',kernel_initializer= initializers.he_normal(seed=None))(CNN_hidden4)
CNN_hidden5 = Dropout(0.5)(CNN_hidden4)

CNN_hidden6 = Dense(b_f*2, activation= 'relu',kernel_initializer= initializers.he_normal(seed=None))(CNN_hidden5)
CNN_hidden6 = Dropout(0.3)(CNN_hidden6)

pred_y = Dense(NUM_CLASS,activation='softmax',kernel_initializer= initializers.he_normal(seed=None))(CNN_hidden6)


concat = Concatenate(axis=-1)([encoder_hidden4,pred_y])


# Latent Represenatation Distribution, P(z)
z_mean = Dense(dim_representation, activation='linear', 
                          kernel_initializer= initializers.he_normal(seed=None))(concat)
z_std_sq_log = Dense(dim_representation, activation='linear', 
                          kernel_initializer= initializers.he_normal(seed=None))(concat)

# Sampling z from P(z)
def sample_z(args):
    mu, std_sq_log = args
    epsilon = K.random_normal(shape=(K.shape(mu)[0], dim_representation), mean=0., stddev=1.)
    z = mu + epsilon * K.sqrt( K.exp(std_sq_log)) 
    return z

z = Lambda(sample_z)([z_mean, z_std_sq_log])
concat2 = Concatenate(axis=-1)([z,pred_y])



# DECODER
# print(K.int_shape(encoder_hidden4)[1])
decoder_hidden0 = Dense(K.int_shape(encoder_hidden4)[1], activation='relu', kernel_initializer= initializers.he_normal(seed=None))(concat2)
decoder_hidden0 = Reshape(K.int_shape(encoder_hidden3)[1:])(decoder_hidden0)

decoder_hidden1 = Conv2DTranspose(filters = b_f*4, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(decoder_hidden0)
decoder_hidden1 = BatchNormalization()(decoder_hidden1)
decoder_hidden1 = Activation('relu')(decoder_hidden1)

decoder_hidden2 = Conv2DTranspose(filters = b_f*2, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(decoder_hidden1)
decoder_hidden2 = BatchNormalization()(decoder_hidden2)
decoder_hidden2 = Activation('relu')(decoder_hidden2)

decoder_hidden3 = Conv2DTranspose(filters = b_f, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(decoder_hidden2)
decoder_hidden3 = BatchNormalization()(decoder_hidden3)
decoder_hidden3 = Activation('relu')(decoder_hidden3)

decoder_hidden4 = Conv2D(filters = 3, kernel_size= 1, strides = (1,1), padding='valid', kernel_initializer = 'he_normal')(decoder_hidden3)
decoder_hidden4 = Activation('sigmoid')(decoder_hidden4)
# MODEL
VAE = Model(inputs=[input_vae,input_CNN],outputs= [decoder_hidden4,pred_y])
print(VAE.summary())




# Encoder Model
Encoder = Model(inputs = [input_vae,input_CNN], outputs = [z_mean, z_std_sq_log])
no_of_encoder_layers = len(Encoder.layers)
no_of_vae_layers = len(VAE.layers)
# print("im here")
# print(VAE.layers[no_of_encoder_layers+1])

# Decoder Model
# decoder_input = Input(shape=(dim_representation+NUM_CLASS,))
# decoder_hidden = VAE.layers[no_of_encoder_layers+2](decoder_input)

# for i in np.arange(no_of_encoder_layers+2 , no_of_vae_layers-1):
    # decoder_hidden = VAE.layers[i](decoder_hidden)
# decoder_hidden = VAE.layers[no_of_vae_layers-1](decoder_hidden)
# Decoder = Model(decoder_input,decoder_hidden )


# Optimizer for Training Neural Network
optimizer_ = RMSprop(lr=0.0001) # Best learning rate = 0.00001

# Compiling the VAE for Training.
VAE.compile(optimizer=optimizer_, loss = [vae_loss,classification_loss], loss_weights = [1.,1.], metrics = {'decoder_hidden4':recon_error,'pred_y':'accuracy'})
H = VAE.fit([X_train,X_train], [X_train,y_train], epochs=20, batch_size=128, validation_data=([X_test,X_test],[X_test,y_test]), verbose=1)
# VAE.load_weights('vae_weights.h5')

choice = np.random.choice(X_test.shape[0],100)
Images = X_test[choice,:,:,:]
# z_mu, z_std = Encoder.predict(Images)
# eps = np.random.normal(0,1, (Images.shape[0], dim_representation))
# z = z_mu + eps * np.sqrt( np.exp(z_std))
prediction = VAE.predict([Images,Images])
Reconstructed = prediction[0]
y_predict = prediction[1]



def plotImages(images,name):
    largeImage = np.zeros((320,320,3))
    for i in range(100):
        h = int(i/10)
        w = i % 10
        largeImage[h*32:(h+1)*32,w*32:(w+1)*32,:] = images[i,:,:,:]
    imsave(name,largeImage)

plotImages(Images,"original.jpg")
plotImages(Reconstructed,"recon_dim"+str(dim_representation)+".jpg")

#training acc

prediction = VAE.predict([X_train,X_train])
y_predict = prediction[1]

num_correct = 0.

for i in range(y_predict.shape[0]):
    predict_label = np.argmax(y_predict[i,:])
    true_label = np.argmax(y_train[i,:])
    if predict_label == true_label:
        num_correct += 1

print("training acc "+str(num_correct/y_predict.shape[0]))


prediction = VAE.predict([X_test,X_test])
y_predict = prediction[1]

num_correct = 0.

for i in range(y_predict.shape[0]):
    predict_label = np.argmax(y_predict[i,:])
    true_label = np.argmax(y_test[i,:])
    if predict_label == true_label:
        num_correct += 1

print("testing acc "+str(num_correct/y_predict.shape[0]))

X_predict = prediction[0]

def RMSE(x_pred,x_true):
    return np.sqrt(np.sum(np.power((x_pred-x_true),2))/(32*32*3))


rmse = 0.
for i in range(X_predict.shape[0]):
    rmse += RMSE(X_predict[i,:,:,:],X_test[i,:,:,:])

rmse = rmse/X_predict.shape[0]
    
print("RMSE")
print(rmse)
