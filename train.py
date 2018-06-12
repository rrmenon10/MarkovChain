__author__ = 'Rakesh R Menon'

import numpy as np
import tensorflow as tf
import tf_utils
import os
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_trainable_state(state, chain_length):
	'''

		Function : Gives the state representation for the Markov Chain as mentioned in the BootstrappedDQN paper.

		Inputs:

			state [Int]: The current state in the Markov Chain.
			chain_length [Int]: The length of the chain.

		Outputs:

			state_rep [Np.Array]: A numpy array with state representation as mentioned in the BootstrappedDQN paper.
	'''

	state_rep = np.zeros((1, chain_length))
	state_rep[0, :state] = 1
	return state_rep

def train(args, env, agent, scope="dqn"):

	'''
		Note : The BootstrappedDQN paper mentions that the algorithm runs for N+9 steps and that the task is considered solved if it gets a return of 10 in the episode.
		To align with these conditions, we have used maximum number of steps as N+7, i.e., N-2 steps till the first reward of +1.0 and then 9 steps more.

		Function : Train the agent on the Markov Chain environment.

		Inputs:

			args [ArgumentParser]: List of arguments that were taken as input in main. Some of the arguments are used in train.
			env [Environment object]: A class object for the Markov Chain environment.
			agent [DQN object]: A class object for the DQN agent.
			scope [String]: Scope name for variables in tensorflow.

	'''

	agent.sess.run(tf.global_variables_initializer())

	#Retrieving operations required for updating target network
	update_target_ops = tf_utils.update_vars_op(scope_from=scope, scope_to="target_"+scope)
	agent.sess.run(update_target_ops)

	#Epsilon annealing
	agent.epsilon = args.start_epsilon
	eps_diff = (args.start_epsilon-args.end_epsilon)/args.anneal_time

	num_episodes = 0
	solved = 0
	tot_steps = 0
	for num_episodes in tqdm(range(2*args.num_episodes)):

		num_steps = 0
		done = False
		state = env.reset()
		if (num_episodes+1)%2==0:			# Training or Testing
			#Testing
			returns = 0.00
			while not done:

				state = get_trainable_state(state, args.chain_length)
				action = agent.act(state)
				next_state, reward = env.step(action)

				num_steps += 1
				returns += reward
				done = False if num_steps < (args.chain_length + 7) else True
				state = next_state

			tqdm.write('%f'%(returns))
			#Counting number of successive times in which the task was solved.
			if int(returns)==10:
				tqdm.write('Solved episode %d!'%(solved+1))
				solved += 1
			else:
				solved = 0

			if solved==100:
				print('Solved! Number of Episodes taken= %d'%((num_episodes-1)/2-100))
				break
		else:
			#Training
			while not done:

				state = get_trainable_state(state, args.chain_length)
				action = agent.act(state)
				next_state, reward = env.step(action)

				num_steps += 1
				tot_steps += 1
				done = True if num_steps >= (args.chain_length + 7) else False

				#Store transition in the replay memory
				agent.mem.store_sample(state.reshape([args.chain_length, ]), action, reward, get_trainable_state(next_state, args.chain_length).reshape([args.chain_length, ]), done)
				state = next_state

				#Update target network. Note : tot_steps used here instead of num_steps
				if (tot_steps+1)%args.update_target==0:
					agent.sess.run(update_target_ops)

				#Perform learning after args.start_learn number of episodes
				if num_episodes/2 < args.start_learn:
					continue

				agent.learn(args.minibatch_size)

			#Anneal epsilon after each episode of training
			agent.epsilon -= eps_diff

	if num_episodes==args.num_episodes and solved!=100:
		print('Unsolved!')