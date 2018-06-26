__author__ = 'Rakesh R Menon'

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import argparse
import mdp
import replay_memory
import model
import tensorflow as tf
from train import train

def main():

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--chain-length', help='Markov Chain length', type=int, default=3)
	parser.add_argument('--num-episodes', help='Number of episodes of training to run', type=int, default=2000)
	parser.add_argument('--start-learn', help='Number of episodes after which learning should start', type=int, default=5)
	parser.add_argument('--replay-size', help='Replay memory size', type=int, default=int(1e4))
	parser.add_argument('--minibatch-size', help='Minibatch size for training', type=int, default=32)
	parser.add_argument('--lr', help='Learning rate for Adam', type=float, default=1e-3)
	parser.add_argument('--update-target', help='Target network updation frequency', type=int, default=100)
	parser.add_argument('--start-epsilon', help='Epsilon-greedy (start)', type=float, default=1)
	parser.add_argument('--end-epsilon', help='Epsilon-greedy (end)', type=float, default=0.1)
	parser.add_argument('--anneal-time', help='Number of episodes to anneal epsilon over', type=int, default=100)

	args = parser.parse_args()
	sess = tf.Session()
	sess.__enter__()
	env = mdp.MarkovChain(chain_length=args.chain_length)
	agent = model.dqn(
		input_size=env.chain_length,
		output_size=env.action_dim,
		hidden_layers=[16, 16],
		activation_fn="relu",
		norm="layer",
		learning_rate=args.lr,
		scope_name="dqn")
	agent.mem = replay_memory.ReplayBuffer(max_samples=args.replay_size)

	train(args, env, agent, scope="dqn")

if __name__=='__main__':
	main()