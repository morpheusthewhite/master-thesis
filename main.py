import argparse

from polarmine.graph import PolarizationGraph
from polarmine.collectors.reddit_collector import RedditCollector
from polarmine.collectors.twitter_collector import TwitterCollector


parser = argparse.ArgumentParser(description='Polarmine')

save_load_group = parser.add_mutually_exclusive_group()
save_load_group.add_argument('--dump', '-d', type=str, default=None, metavar='dump',
                             help='dump the mined data at the given path')
save_load_group.add_argument('--load', '-l', type=str, default=None, metavar='load',
                             help='load the mined data at the given path')
# Model params
#  parser.add_argument('--model-name', '-mn', type=str, default='vae', metavar='model_name',
#                      help='model name: vae, vamp', choices=['vae', 'vamp', 'hvae'])
#  parser.add_argument('-C', '--pseudo-inputs', type=int, default=500, metavar='C', dest='C',
#                      help='number of pseudo-inputs with vamp prior')
#  parser.add_argument('-D', type=int, default=40, metavar='D',
#                      help='number of stochastic hidden units, i.e. z size (same for z1 and z2 with HVAE)')
#  parser.add_argument('--dataset', '-ds', type=str, default='mnist', metavar='dataset',
#                      help='used dataset: mnist, frey', choices=['mnist', 'frey', 'fashion'])
#  # Training params
#  parser.add_argument('--epochs', '-e', type=int, default=1, metavar='epochs',
#                      help='number of epochs')
#  parser.add_argument('-bs', '--batch-size', type=int, default=100, metavar='batch_size',
#                      help='size of training mini-batch')
#  parser.add_argument('-L', type=int, default=1, metavar='L',
#                      help='number of MC samples')
#  parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, metavar='lr', dest='lr',
#                      help='learning rate')
#  parser.add_argument('-wu', '--warm-up', type=int, default=0, metavar='warmup', dest='warmup',
#                      help='number of warmup epochs')
#  parser.add_argument('--max-beta', type=float, default=1., metavar='max_beta',
#                      help='maximum value of the regularization loss coefficient')
#  # Debugging params
#  parser.add_argument('-tb', '--tensorboard', action='store_true', dest='tb',
#                      help='save training log in ./ for tensorboard inspection')
#  parser.set_defaults(tb=False)
#  parser.add_argument('-d', '--debug', action='store_true', dest='debug',
#                      help='show images')
#  parser.set_defaults(debug=False)

args = parser.parse_args()


def main():

    if args.load is not None:
        graph = PolarizationGraph.from_file(args.load)
    else:
        # mine data and store it
        reddit_collector = RedditCollector()
        contents = list(reddit_collector.collect(1, limit=10, cross=False))

        graph = PolarizationGraph(contents)

        if args.dump is not None:
            graph.dump(args.dump)

    graph.draw()


if __name__ == "__main__":
    main()
