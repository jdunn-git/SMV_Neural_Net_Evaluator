# SMV_Neural_Net_Evaluator

## Description
This repo holds code to evaluator and train a neural net in Python, as well as an smv-based representation of the neural net. This can be used to verify the results of training the neural net. Please note that this is verifying for robustness, not validating for correctness.

The neural net used by this repo has drawn inspiration from many sources areound the web, such as:

https://visualstudiomagazine.com/articles/2019/07/01/modern-neural-networks.aspx

and

https://realpython.com/python-ai-neural-network/

It is a ReLU-based neural net with 3 input nodes, 2 hidden nodes, and one output node. Below is  a diagram of the neural net being used. For ease of translation into the smv program, I only allowed the hidden-to-output weights to be trained. The input-to-hidden weights and every node biased stayed constant. 
![NN SMV Sequence Diagram.png](https://github.com/jdunn-git/SMV_Neural_Net_Evaluator/blob/main/Neural%20Net.png "Neural net that is trained and evaluated by this repo")


The neural net certainly isn't perfect, but for the verification problem I was attempting to solve it didn't need to be. Instead, I merely needed to meet two criteria:
1. I needed to be able to represent the neural net evaluation in the nuXmv model checker.
2. I needed to prove that training the imperfect neural net could produce provable results. This is merely a verification of the process, not a validation of the neural net.

The main program is smv_evaluator.py. This will create and train the neural net, and continue training until a couple of thresholds have been met:
1. The neural net output given an input must be below a certain threshold.
2. Given adjustments to the input of the neural net, the output must differ by no more than a certain threshold.

The thresholds, inputs, and other configurable values can be found in smv.yaml.

Inside the smv/ folder, there is a file called nn.smv.template. This stores a smv program that has value tags, which will be replaced by the python program when it is running. Using this mechanism, I have been able to generate multiple smv files and evaluate them, all from within the smv_evaluator.py program.

The program will continue running until either both provable criteria have been met, or until it has trained the neural net beyond a maximum threshold. This is to prevent an infinite loop, since it's entirely possible that the results of the training can't be proven against the configuration values.

![NN SMV Sequence Diagram.png](https://github.com/jdunn-git/SMV_Neural_Net_Evaluator/blob/main/NN%20SMV%20Sequence%20Diagram.png "Sequence diagram of the smv_evaluator program")

## Running 
Prerequisites can be installed by running
>$ pip install -r requirements.txt

The program can run by the command
>$ cd smv_evaluator/
>
>$ python smv_evaluator.py


## Evaluation Data
The program with base configuration generated a neural net that will product the following data:
Given two inputs (200, 0, 20) and (200, 0, 15):

| Training Iterations | Hidden-Output Weight 1 | Hidden-Output Weight 2 | (200, 0, 20) Output | (200, 0, 15) Output | Difference |
|---|---|---|---|---|---|
|0|10|10|212|162|50|
|1|11|9|191|146|45|
|2|11|9|191|146|45|
|3|12|8|170|130|40|
|4|12|8|170|130|40|
|5|13|7|149|114|35|
|6|13|7|149|114|35|
|7|13|7|149|114|35|
|8|14|6|128|98|30|
|9|14|6|128|98|30|
|10|14|6|128|98|30|
|11|14|6|128|98|30|
|12|15|5|107|82|25|
|13|15|5|107|82|25|
|14|15|5|107|82|25|
|15|15|5|107|82|25|
|16|16|4|86|66|20|
|17|16|4|86|66|20|
|18|16|4|86|66|20|
|19|16|4|86|66|20|
|20|16|4|86|66|20|
|21|16|4|86|66|20|
|22|17|3|65|50|15|
|23|17|3|65|50|15|
|24|17|3|65|50|15|
|25|17|3|65|50|15|

![NN Weight Adjustments.png](https://github.com/jdunn-git/SMV_Neural_Net_Evaluator/blob/main/NN%20Weight%20Adjustments.png "Differences in the neural net weights over the course of training")
![NN Output Changes and Differences.png](https://github.com/jdunn-git/SMV_Neural_Net_Evaluator/blob/main/NN%20Output%20Changes%20and%20Differences.png "Differences in the neural net output for the two inputs over the course of training")
