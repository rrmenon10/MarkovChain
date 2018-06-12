__author__ = 'Rakesh R Menon'

class MarkovChain:

	def __init__(self, chain_length):

		if chain_length <=2:
			raise ValueError('Please provide Markov Chain length > 2 for task.')

		self.current_state = 2
		self.goal_state = self.chain_length = chain_length
		self.action_dim = 2

	def reset(self):

		self.current_state = 2
		return self.current_state
	

	def step(self, action):

		if action == 0:
			if self.current_state != 1:
				self.current_state -= 1

		elif action == 1:
			if self.current_state != self.goal_state:
				self.current_state += 1

		else:
			raise ValueError("Action out of bounds")

		if self.current_state == self.goal_state:
			return self.current_state, 1.00

		elif self.current_state == 1:
			return self.current_state, 0.001

		else:
			return self.current_state, 0.00