#
# neural_net.py contains the Neural Net that the smt_evaluator will use to train and evaluate. It
#   currently uses a ReLU activation method. While many parts of the neural net have been built to
#	be easily resizable, some aspects of the training and evaluation function are specific for the
#	3-2-1 network that the smt_evaluator is using in order to stay more in line with the smt program
#	that will be evaluated from smt_evaluator.
#

import utils
import numpy

class NeuralNetwork:
	def __init__(self, num_input, num_hidden, num_output):
		self.ni = num_input
		self.nh = num_hidden
		self.no = num_output

		self.i_nodes = numpy.zeros(shape=[self.ni], dtype=numpy.float32)
		self.h_nodes = numpy.zeros(shape=[self.nh], dtype=numpy.float32)
		self.o_nodes = numpy.zeros(shape=[self.no], dtype=numpy.float32)

		self.ih_weights = numpy.zeros(shape=[self.ni,self.nh], dtype=numpy.float32)
		self.ho_weights = numpy.zeros(shape=[self.nh,self.no], dtype=numpy.float32)
		self.h_biases = numpy.zeros(shape=[self.nh], dtype=numpy.float32)
		self.o_biases = numpy.zeros(shape=[self.no], dtype=numpy.float32)

		self.h_sums = numpy.zeros(shape=[self.nh], dtype=numpy.float32)
		self.o_sums = numpy.zeros(shape=[self.no], dtype=numpy.float32)

		self.training_iterations = 0

	def set_weights(self, weights):
		index = 0
		for i in range(self.ni):
			for j in range(self.nh):
				self.ih_weights[i,j] = weights[index]
				index += 1

		for j in range(self.nh):
			self.h_biases[j] = weights[index]
			index += 1

		for j in range(self.nh):
			for k in range(self.no):
				self.ho_weights[j,k] = weights[index]
				index += 1

		for k in range(self.no):
			self.o_biases[k] = weights[index]
			index += 1

	def get_output_weights(self):
		output_weights = []
		for weight in self.ho_weights:
			output_weights.append(weight[0])
		return output_weights

	def eval(self, x_values, training=False):
		# Reset h_sums and o_sums
		self.h_sums = numpy.zeros(shape=[self.nh], dtype=numpy.float32)
		self.o_sums = numpy.zeros(shape=[self.no], dtype=numpy.float32)

		self.i_nodes = x_values  # by ref

		# Evaluate hidden nodes
		for j in range(self.nh):
			for i in range(self.ni):
				if not training and utils.debug:
					print(f"\nih_weights[{i}, {j}]: {self.ih_weights[i,j]}")
				self.h_sums[j] += self.i_nodes[i] *self.ih_weights[i,j]

			# Add the bias to each hidden node
			self.h_sums[j] += self.h_biases[j]

		# Alternative way to handle the above
		#self.h_sums = numpy.dot(self.i_nodes, self.ih_weights)
		#self.h_sums += self.h_biases

		if not training and utils.debug:
			print("\nPre-activation hidden node values: ")
			utils.show_vec(self.h_sums, 8, 4, len(self.h_sums))

		# Activate ReLU function on
		for j in range(self.nh):
			self.h_nodes[j] = utils.relu(self.h_sums[j])

		for k in range(self.no):
			for j in range(self.nh):
				if not training and utils.debug:
					print(f"\nh_nodes[{j}]: {self.h_nodes[j]}")
					print(f"\nho_weights[{j}, {k}]: {self.ho_weights[j, k]}")
				self.o_sums[k] += self.h_nodes[j] * self.ho_weights[j,k]
			self.o_sums[k] += self.o_biases[k]
		if not training and utils.debug:
			print("\nPre-activation output bias values: ")
			utils.show_vec(self.o_biases, 8, 4, len(self.o_biases))
			print("\nPre-activation output node values: ")
			utils.show_vec(self.o_sums, 8, 4, len(self.o_sums))

		for k in range(self.no):
			if training:
				self.o_nodes[k] = self.o_sums[k]
			else:
				# Only output value between 0 and 255
				if self.o_sums[k] > 255.0:
					self.o_nodes[k] = numpy.float32(255)
				elif self.o_sums[k] < 0.0:
					self.o_nodes[k] = numpy.float32(0)
				else:
					self.o_nodes[k] = self.o_sums[k]
				self.o_nodes[k] += self.o_biases[k]

		result = numpy.zeros(shape=self.no, dtype=numpy.float32)
		for k in range(self.no):
			result[k] = self.o_nodes[k]

		return result

	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in range(number_of_training_iterations):
			# Pass the training set through our neural network (a single neuron).
			outputs = []
			for input in training_set_inputs:
				outputs.append(self.eval(input, training=True))

			# Calculate the error (The difference between the desired output
			# and the predicted output).
			error_sets = []
			i = 0
			for output in outputs:  # todo clean this up
				output = outputs[i]
				training_set_output = training_set_outputs[i]
				j = 0
				error_set = []
				while j < len(output):
					o = output[j]
					t = training_set_output[j]
					e = o - t
					error_set.append(training_set_output[j] - output[j])
					j += 1
				error_sets.append(error_set)
				i += 1

			# Generate the adjustments to correct for each error value from the training data
			i = 0
			while i < len(outputs):
				output_set = outputs[i]
				error_set = error_sets[i]
				adjustments = [0] * len(error_sets)
				j = 0
				while j < len(output_set):

					output = output_set[j]
					error = error_set[j]

					derivative = utils.relu_derivative(output)
					adj = numpy.float32(error * derivative)
					adjustments[j] = adj
					j += 1
				i += 1


			# Try only adjusting one of the weights depending on the adjustment
			for adj in adjustments:
				modified_adj = adj * 0.001
				if modified_adj > 0:
					old_weight = self.ho_weights[0, 0]
					tmp_weight = old_weight + modified_adj
					new_weight = numpy.float32(tmp_weight)
					self.ho_weights[0, 0] = new_weight
					old_weight = self.ho_weights[1, 0]
					tmp_weight = old_weight + (modified_adj * -1)
					new_weight = numpy.float32(tmp_weight)
					self.ho_weights[1, 0] = new_weight
				else:
					old_weight = self.ho_weights[0, 0]
					tmp_weight = old_weight + (modified_adj * -1)
					new_weight = numpy.float32(tmp_weight)
					self.ho_weights[0, 0] = new_weight
					old_weight = self.ho_weights[1, 0]
					tmp_weight = old_weight + modified_adj
					new_weight = numpy.float32(tmp_weight)
					self.ho_weights[1, 0] = new_weight



			self.training_iterations += 1





