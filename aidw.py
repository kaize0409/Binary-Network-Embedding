from keras.layers import Dense, Input, noise, LeakyReLU
from keras.layers.merge import concatenate, multiply
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop
from keras.regularizers import L1L2
import keras.backend as K

import dataset
import node2vec
import numpy as np
import scipy.io as sio




def cross_entropy_loss(y_true, y_pred):
    return - K.mean(K.log(K.sigmoid(K.clip(K.sum(y_pred, axis = 1)*y_true, -6, 6))))


def context_preserving(latent_dim):
    node_rep = Input(shape=(latent_dim, ), name='node_rep')
    context_rep = Input(shape=(latent_dim, ), name='context_rep')
    sim = multiply([node_rep, context_rep])
    return Model(inputs=[node_rep, context_rep], outputs=sim, name='context_aware')


def encoder(node_num, hidden_layers, hidden_neurons):
    x = Input(shape=(node_num,))
    encoded = noise.GaussianNoise(0.2)(x)
    for i in range(hidden_layers):
        encoded = Dense(hidden_neurons[i])(encoded)
        encoded = LeakyReLU(0.2)(encoded)
        BatchNormalization()
        encoded = noise.GaussianNoise(0.2)(encoded)
    return Model(inputs=x, outputs=encoded)


def model_discriminator(latent_dim, output_dim=2, hidden_dim=512, reg=lambda: L1L2(1e-7, 1e-7)):
    z = Input((latent_dim,))
    h = Dense(hidden_dim, kernel_regularizer=reg())(z)
    h = LeakyReLU(0.2)(h)
    BatchNormalization()
    h = Dense(hidden_dim, kernel_regularizer=reg())(h)
    h = LeakyReLU(0.2)(h)
    BatchNormalization()
    y = Dense(output_dim, activation="softmax", kernel_regularizer=reg())(h)
    return Model(z, y)




def AIDW(args):

    # read data and define batch generator
    nx_G = dataset.read_graph(args)
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    adj_mat = dataset.load_data(args.input_adj)
    network = dataset.load_data(args.input_ppmi)
    walk_generator = dataset.contextSampling(walks, adj_mat, args)

    # read parameters
    dataset_name = args.input_ppmi.split('/')[1].split('.')[0]
    hidden_layers = args.hidden_layers
    neurons = args.hidden_neurons
    neurons = neurons.split('/')
    hidden_neurons = []
    for i in range(len(neurons)):
        hidden_neurons.append(int(neurons[i]))
    node_num = adj_mat.shape[0]
    latent_dim = hidden_neurons[-1]

    # build the graph
    # embeddings (x ->z)
    encoder_node = encoder(node_num, hidden_layers, hidden_neurons)
    encoder_context = encoder(node_num, hidden_layers, hidden_neurons)

    # context preserving
    context_prediction = context_preserving(latent_dim)

    # constructing context preserving model
    node = encoder_node.inputs[0]
    context = encoder_context.inputs[0]
    node_rep = encoder_node(node)
    context_rep = encoder_context(context)
    sim = context_prediction([node_rep, context_rep])
    context_model = Model(inputs=[node, context], outputs=sim)
    context_model.compile(optimizer=RMSprop(lr=args.lr), loss=cross_entropy_loss)

    # discriminator (z -> y)
    discriminator = model_discriminator(latent_dim)
    discriminator.compile(optimizer=RMSprop(lr=args.lr), loss='categorical_crossentropy')

    # generator
    x = encoder_node.inputs[0]
    z = encoder_node(x)
    y_fake = discriminator(z)
    gan = Model(inputs=x, outputs=y_fake)
    gan.compile(optimizer=RMSprop(lr=args.lr), loss='mse')

    epoch_gen_loss = []
    epoch_disc_loss = []
    epoch_context_loss = []
    index = 0

    while True:
        index = index + 1
        l_nodes, r_nodes, labels = next(walk_generator)
        batchsize = l_nodes.shape[0]
        left_batch = network[l_nodes]
        right_batch = network[r_nodes]
        data_batch = np.concatenate([left_batch, right_batch], axis=0)

        for t in range(args.T0):
            epoch_context_loss.append(context_model.train_on_batch([left_batch, right_batch], labels))

        # the updating of the discriminator
        noise = np.random.uniform(-1.0, 1.0, [2*batchsize, latent_dim])
        z_batch = encoder_node.predict(data_batch)
        X = np.concatenate((noise, z_batch))
        y_dis = np.zeros([4 * batchsize, 2])
        y_dis[0:2*batchsize, 1] = 1
        y_dis[2*batchsize:, 0] = 1 

        for t in range(args.T1):
            # clip weights
            discriminator.trainable = True
            weights = [np.clip(w, -0.01, 0.01) for w in discriminator.get_weights()]
            discriminator.set_weights(weights)
            epoch_disc_loss.append(discriminator.train_on_batch(X, y_dis))

        # the updating of the generator
        y_fake = np.zeros([2*batchsize, 2])
        y_fake[:, 1] = 1
        for t in range(args.T2):
            discriminator.trainable = False
            epoch_gen_loss.append(gan.train_on_batch(data_batch, y_fake))


        if (index)%(200) == 0:
            print '\nTraining loss for index {}:'.format(index)
            context_loss = np.mean(np.array(epoch_context_loss[-50:]), axis=0)
            dis_loss = np.mean(np.array(epoch_disc_loss[-50:]), axis=0)
            gen_loss = np.mean(np.array(epoch_gen_loss[-50:]), axis=0)
            print 'AutoE-{} Dis-{} Gen-{}'.format(context_loss, dis_loss, gen_loss)

            rep = encoder_node.predict(network)
            rep_file = 'output/{}-rep-{}.mat'.format(dataset_name, str((index)/200))
            sio.savemat(rep_file, {'rep':rep})
            sio.savemat(args.rep, {'rep':rep})
