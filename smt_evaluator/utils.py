#
# utils.py contains utility objects and methods for smt_evaluator and neural_net. Some functions
#	like the leaky activation function are not used at this time but remain here in case someone
#	wants to test with them.
#

import numpy

debug = False
MAX_RGB = 256

class NN_Results:
	def __init__(self, nn_output=0, nn_input=None, nn_ho_weights=None, training_iterations=0):
		if nn_ho_weights is None:
			nn_ho_weights = []
		if nn_input is None:
			nn_input = []

		self.nn_output = nn_output
		self.nn_input = nn_input
		self.nn_ho_weights = nn_ho_weights
		self.training_iterations = training_iterations

	def set_nn_output(self, num):
		self.nn_output = num

	def set_nn_input(self, input_array):
		self.nn_input = input_array

	def set_ho_weights(self, nn_ho_weights_array):
		self.nn_ho_weights = nn_ho_weights_array

	def set_training_iterations(self, training_iterations):
		self.training_iterations = training_iterations

	def print(self):
		print("------- Results of neural net testing: -------")
		print(f"\tAfter {self.training_iterations} training iterations...")
		print(f"\tinput: {self.nn_input}")
		print(f"\thidden-output weights: {self.nn_ho_weights}")
		print(f"\toutput: {self.nn_output}")
		print("----------------------------------------------")

def show_vec(v, wid, dec, vals_line):
	fmt = "% " + str(wid) + "." + str(dec) + "f"  # like % 8.4f
	for i in range(len(v)):
		if i > 0 and i % vals_line == 0: print("")
		print(fmt % v[i] + " ", end="")
	print("")

def show_matrix(m, wid, dec):
	fmt = "% " + str(wid) + "." + str(dec) + "f"  # like % 8.4f
	for i in range(len(m)):
		for j in range(len(m[i])):
			x = m[i,j]
			print(fmt % x + " ", end="")
		print("")

def relu(x):
	#print(f"relu({x})")
	if x < 0.0:
		return 0.0
	else:
		return x

def relu_derivative(x):
	if x < 0.0:
		return 0.0
	else:
		return 1.0

def leaky(x):
	if x < 0.0:
		return 0.01 * x
	else:
		return x

def leaky_derivative(x):
	if x < 0.0:
		return 0.01
	else:
		return 1.0

def softmax(vec):
	n = len(vec)
	result = numpy.zeros(n, dtype=numpy.float32)
	mx = numpy.max(vec)
	divisor = 0.0
	for k in range(n):
		divisor += numpy.exp(vec[k] - mx)
	for k in range(n):
		result[k] = numpy.exp(vec[k] - mx) / divisor
	return result

def generate_array_from_color(color, makearray=True):
	standardized_color = []
	for rgb in color:
		# This will standardize colors from the range -128 to 128
		#standardized_color.append(rgb - (MAX_RGB/2))

		# This will normalize the color between 0 and 1
		#standardized_color.append(rgb / numpy.float32(MAX_RGB))

		# This will just create the color as an array
		standardized_color.append(rgb)
	if makearray:
		return numpy.array(standardized_color, dtype=numpy.float32)
	else:
		return standardized_color

def normalize_rgb(rgb_tuples):
	normalized_rgb_tuples = []
	for rgb_tuple in rgb_tuples:
		normalized_rgb = []
		for color in rgb_tuple:
			normalized_rgb.append(numpy.float32(color / MAX_RGB))

		normalized_rgb_tuples.append(normalized_rgb)

	return normalized_rgb_tuples

# Ensures x is restrained to be restrained between a min and a max
def restain(x, max, min):
	if x > max:
		return max
	if x < min:
		return min
	return x