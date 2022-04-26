This repo holds code to evaluator and train a neural net in Python, as well as an smv-based representation of the neural net. This can be used to verify the results of training the neural net.

The nerual net used by this repo has drawn inspiration from many sources areound the web, such as:

https://visualstudiomagazine.com/articles/2019/07/01/modern-neural-networks.aspx

and

https://realpython.com/python-ai-neural-network/


The neural net certainly isn't perfect, but for the verification problem I was attempting to solve it didn't need to be. Instead, I merely needed to meet two criteria:
1. I needed to be able to represent the neural net evaluation in the nuXmv model checker.
2. I needed to prove that training the imperfect neural net could produce provable results. This is merely a verification of the process, not a validation of the neural net.

The main program is smt_evaluator.py. This will create and train the neural net, and continue training until a couple of thresholds have been met:
1. The neural net output given an input must be below a certain threshold.
2. Given adjustments to the input of the neural net, the output must differ by no more than a certain threshold.

The thresholds, inputs, and other configurable values can be found in smv.yaml.

Inside the smt/ folder, there is a file called nn.smv.template. This stores a smv program that has value tags, which will be replaced by the python program when it is running. Using this mechanism, I have been able to generate multiple smv files and evaluate them, all from within the smv_evaluator.py program.

The program will continue running until either both provable criteria have been met, or until it has trained the neural net beyond a maximum threshold. This is to prevent an infinite loop, since it's entirely possible that the results of the training can't be proven against the configuration values.