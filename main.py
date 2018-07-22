import argparse
import aidw




def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Adversarial Inductive DeepWalk.")

    # for input
    parser.add_argument('--input_edgelist', nargs='?', default='input/citeseer-edgelist.txt', help='Input graph path')
    parser.add_argument('--input_ppmi', nargs='?', default='input/citeseer-PPMI-4.mat', help='Input PPMI')
    parser.add_argument('--input_adj', nargs='?', default='input/citeseer-undirected.mat', help='Input adjMat')
    parser.add_argument('--rep', nargs='?', default='output/citeseer-rep.mat', help='Embeddings path')

    # for random walk
    parser.add_argument('--walk_length', type=int, default=10,help='Length of walk per source.')
    parser.add_argument('--num_walks', type=int, default=10, help='Number of walks per source.')
    parser.add_argument('--window_size', type=int, default=10, help='Context size for optimization.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
    parser.add_argument('--K', type=int, default=5, help='Number of negative pairs for each positive pair.')
    parser.add_argument('--p', type=float, default=1, help='Return hyperparameter.')
    parser.add_argument('--q', type=float, default=1, help='Input hyperparameter.')
    parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)
    parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    # for GAN
    parser.add_argument('--hidden_layers', type=int, default=1, help='Number of hidden layers.')
    parser.add_argument('--hidden_neurons', nargs='?', default='128', help='Hidden neurons')

    # for training
    parser.add_argument('--T0', type=int, default=1, help='Context loss iteration times.')
    parser.add_argument('--T1', type=int, default=1, help='Discriminator iteration times.')
    parser.add_argument('--T2', type=int, default=1, help='Generator iteration times.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')

    # for test
    parser.add_argument('--result_file_name', nargs='?', default='output/AIDW-citeseer.txt')

    return parser.parse_args()




def main(args):
    result_file = open(args.result_file_name, 'a')
    settings = (
        'num_walks-{}-walk_length-{}-window_size-{}-K-{}-batch_size-{}-lr-{}\ninput-{}\ninput_ppmi-{}\ninput_adj-{}'
        '\nhidden_neurons-{}\nLeak-0.2'
    ).format(
        str(args.num_walks),
        str(args.walk_length),
        str(args.window_size),
        str(args.K),
        str(args.batch_size),
        str(args.lr),
        args.input_edgelist,
        args.input_ppmi,
        args.input_adj,
        args.hidden_neurons
    )
    result_file.write(settings)
    result_file.close()
    aidw.AIDW(args)




if __name__ == "__main__":
    args = parse_args()
    main(args)
