import numpy as np
import scipy.io as sio
import networkx as nx
import math




class UnigramTable:
    """
    Refer to https://github.com/deborausujono/word2vecpy/blob/master/word2vec.py
    Using degree list to initialize the drawing
    """
    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        norm = sum([math.pow(t, power) for t in vocab]) # Normalizing constant

        table_size = int(1e8) # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print 'Filling unigram table'
        p = 0 # Cumulative probability
        i = 0
        for t in range(vocab_size):
            p += float(math.pow(vocab[t], power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = t
                i += 1
        self.table = table
        print 'Finish filling unigram table'

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]




def load_data(file):
    # load data
    data_mat = sio.loadmat(file)
    data_mat = data_mat['network']
    data_mat = data_mat.toarray()
    return data_mat


def read_graph(args):
    '''
    Reads the input network in networkx.
    From node2vec source code
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input_edgelist, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input_edgelist, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def contextSampling(walks, adj_mat, args):
    '''
    batch generator
    '''
    walks_num = walks.shape[0]
    walks_len = walks.shape[1]
    degree = list(np.sum(adj_mat, 1))
    table = UnigramTable(degree)

    count = 0
    l_nodes = []
    r_nodes = []
    labels = []
    while True:
        for i in range(walks_num):
            for l in range(walks_len):
                for m in range(l-args.window_size, l+args.window_size+1):
                    if m<0 or m>=walks_len: continue
                    l_nodes.append(walks[i, l]-1)
                    r_nodes.append(walks[i, m]-1)
                    labels.append(1.0)
                    # negative samples corresponding to the current positive pair
                    for k in range(args.K):
                        n_neg = table.sample(1)[0]
                        while n_neg==walks[i, l]-1 or n_neg==walks[i, m]-1:
                            n_neg = table.sample(1)[0]

                        l_nodes.append(walks[i, l]-1)
                        r_nodes.append(n_neg)
                        labels.append(-1.0)
                    count = count + 1
                    if count>=args.batch_size:
                        yield np.array(l_nodes, dtype=np.int32), np.array(r_nodes, dtype=np.int32), np.array(labels, dtype=np.float32)
                        l_nodes = []
                        r_nodes = []
                        labels = []
                        count = 0
