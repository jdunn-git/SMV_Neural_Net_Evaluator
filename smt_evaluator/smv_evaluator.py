#
# smv_evaluator.py will create and train a neural net. It will then use the results of the neural net
#   training and evaluation to replace tags in a templatized-smv program, in order to evaluate the
#   neural net from an smv program.
#

import sys
import utils
import neural_net
import numpy
import yaml
import subprocess

def main():
    # print("Begin NN with leaky ReLU IO demo")

    with open("smv.yaml", "r") as stream:
        try:
            parsed_smt = yaml.safe_load(stream)
            print(f"Config: {parsed_smt}")
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    # Create network
    print("\nCreating a 3-2-1 leaky ReLU NN")
    nn = neural_net.NeuralNetwork(3, 2, 1)

    ih_weight_1 = parsed_smt['hidden_1_input_1_weight']
    ih_weight_2 = parsed_smt['hidden_1_input_2_weight']
    ih_weight_3 = parsed_smt['hidden_1_input_3_weight']
    ih_weight_4 = parsed_smt['hidden_2_input_1_weight']
    ih_weight_5 = parsed_smt['hidden_2_input_2_weight']
    ih_weight_6 = parsed_smt['hidden_2_input_3_weight']

    hidden_bias_1 = parsed_smt['hidden_1_bias']
    hidden_bias_2 = parsed_smt['hidden_2_bias']
    output_bias = parsed_smt['output_bias']

    ho_bias_1 = parsed_smt['initial_ho1_weight']
    ho_bias_2 = parsed_smt['initial_ho2_weight']

    # Set weights and biases
    weights = numpy.array([ih_weight_1, ih_weight_2, ih_weight_3, ih_weight_4, ih_weight_5, ih_weight_6,  # ih weights
                           hidden_bias_1, hidden_bias_2,  # h biases
                           ho_bias_1, ho_bias_2,  # ho weights
                           output_bias], dtype=numpy.float32)  # o biases

    print("\nSetting weight and biases")
    utils.show_vec(weights, wid=6, dec=2, vals_line=8)
    nn.set_weights(weights)

    input_color = [parsed_smt['input_1'], parsed_smt['input_2'], parsed_smt['input_3']]
    normalized_color = utils.generate_array_from_color(input_color)

    # Set input
    print("\nSetting inputs to: ")
    utils.show_vec(normalized_color, 6, 2, len(normalized_color))

    # Track all results
    results = []

    # Compute Outputs
    print("\nComputing output values...")
    original_output = nn.eval(normalized_color)
    print("\nOutput values: ")
    utils.show_vec(original_output, 8, 4, len(original_output))

    # Store pre-training output
    output_0 = original_output[0]
    output_weights_0 = nn.get_output_weights()
    result = utils.NN_Results(output_0, normalized_color, output_weights_0, 0)
    results.append(result)

    training_set_inputs = [[230, 10, 25], [20, 60, 230], [30, 210, 210], [240, 240, 130], [230, 170, 50]]
    training_set_outputs = [[10], [95], [80], [50], [35]]
    # training_set_outputs = [[70], [95], [80], [90], [98]]

    normalized_inputs = []
    for inputs in training_set_inputs:
        normalized_inputs.append(utils.generate_array_from_color(inputs, makearray=False))

    normalized_inputs = numpy.array(normalized_inputs)

    normalized_outputs = numpy.array(training_set_outputs)

    '''
    #
    # This commented code block will train the neural net and display evaluation results after the
    #   following number of training iterations: 0, 5, 10, 20, 50, 200, 2000
    #   
    
    # Train neural net
    print("\nTraining neural net...")
    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    nn.train(normalized_inputs, normalized_outputs, 5)

    # Compute Outputs
    print("\nComputing output values...")
    original_output = nn.eval(normalized_color)
    print("\nOutput values: ")
    utils.show_vec(original_output, 8, 4, len(original_output))
    # Compute output_5
    output_5 = original_output[0]
    output_weights_5 = nn.get_output_weights()
    result = utils.NN_Results(output_5, normalized_color, output_weights_5, 5)
    results.append(result)

    # Compute output_10 (5 more)
    nn.train(normalized_inputs, normalized_outputs, 5)
    original_output = nn.eval(normalized_color)
    output_10 = original_output[0]
    output_weights_10 = nn.get_output_weights()
    result = utils.NN_Results(output_10, normalized_color, output_weights_10, 10)
    results.append(result)


    # Compute output_20 (10 more)
    nn.train(normalized_inputs, normalized_outputs, 10)
    original_output = nn.eval(normalized_color)
    output_20 = original_output[0]
    output_weights_20 = nn.get_output_weights()
    result = utils.NN_Results(output_20, normalized_color, output_weights_20, 20)
    results.append(result)

    # Compute output_50 (30 more)
    nn.train(normalized_inputs, normalized_outputs, 30)
    original_output = nn.eval(normalized_color)
    output_50 = original_output[0]
    output_weights_50 = nn.get_output_weights()
    result = utils.NN_Results(output_50, normalized_color, output_weights_50, 50)
    results.append(result)

    # Compute output_200 (150 more)
    nn.train(normalized_inputs, normalized_outputs, 150)
    original_output = nn.eval(normalized_color)
    output_200 = original_output[0]
    output_weights_200 = nn.get_output_weights()
    result = utils.NN_Results(output_200, normalized_color, output_weights_200, 200)
    results.append(result)


    nn.train(normalized_inputs, normalized_outputs, 1800)
    original_output = nn.eval(normalized_color)
    output_2000 = original_output[0]
    output_weights_2000 = nn.get_output_weights()
    result = utils.NN_Results(output_10, normalized_color, output_weights_10, 10)
    results.append(result)


    print("Output results after training X times:")
    print(f" 0:    {output_0:.4f}\tweights: {output_weights_0[0]:.4f}, {output_weights_0[1]:.4f}")
    print(f" 5:    {output_5:.4f}\tweights: {output_weights_5[0]:.4f}, {output_weights_5[1]:.4f}")
    print(f" 10:   {output_10:.4f}\tweights: {output_weights_10[0]:.4f}, {output_weights_10[1]:.4f}")
    print(f" 20:   {output_20:.4f}\tweights: {output_weights_20[0]:.4f}, {output_weights_20[1]:.4f}")
    print(f" 50:   {output_50:.4f}\tweights: {output_weights_50[0]:.4f}, {output_weights_50[1]:.4f}")
    print(f" 200:  {output_200:.4f}\tweights: {output_weights_200[0]:.4f}, {output_weights_200[1]:.4f}")
    print(f" 2000: {output_2000:.4f}\tweights: {output_weights_2000[0]:.4f}, {output_weights_2000[1]:.4f}")
    '''

    # Repeatedly train the neural net and retest the smt solver until desired outcome has been reached
    count = 0
    trainings_per_loop = 1
    while True:
        nn.train(normalized_inputs, normalized_outputs, trainings_per_loop)
        output = nn.eval(normalized_color)[0]
        output_weights = nn.get_output_weights()
        result = utils.NN_Results(output, normalized_color, output_weights, trainings_per_loop * count)
        results.append(result)
        result.print()

        #
        # Compute the smt statistics from here after 20 trainings - round to an int
        #
        parsed_smt['hidden_1_output_weight'] = round(result.nn_ho_weights[0])
        parsed_smt['hidden_2_output_weight'] = round(result.nn_ho_weights[1])
        # parsed_smt['output_threshold'] = 100
        # This will use the smt to prove that the result should have been less than 100 now, given the current weights

        # Generate two test files - one for each of the criteria
        # Generate SMV file 1
        # Read in the file
        with open('../smt/nn.smv.template', 'r') as file:
            filedata = file.read()

        # Replace the target string
        for key, val in parsed_smt.items():
            filedata = filedata.replace(f'<<{key}>>', f'{val}')

        # Write the file out again
        with open('../smt/nn_1.smv', 'w') as file:
            file.write(filedata)

        # Generate SMV file 2
        # Read in the file
        with open('../smt/nn.smv.template', 'r') as file:
            filedata = file.read()

        # Replace the target string
        for key, val in parsed_smt.items():
            # Adjust the
            if key == 'input_3':
                new_input_3 = str(int(parsed_smt['input_3']) + int(parsed_smt['input_3_difference']))
                filedata = filedata.replace(f'<<{key}>>', f'{new_input_3}')
            elif key == 'output_threshold':
                new_output_threshold = str(int(output) + int(parsed_smt['output_difference']))
                filedata = filedata.replace(f'<<{key}>>', f'{new_output_threshold}')
            elif key == 'output_symbol':
                filedata = filedata.replace(f'<<{key}>>', f'{parsed_smt["output_difference_symbol"]}')
            else:
                filedata = filedata.replace(f'<<{key}>>', f'{val}')

        # Write the file out again
        with open('../smt/nn_2.smv', 'w') as file:
            file.write(filedata)

        # Run nuXmv program 1
        result = subprocess.run(["../smt/nuxmv/nuxmv.exe", "../smt/nn_1.smv"], capture_output=True, text=True)
        nuxmv_output = result.stdout

        # Evaluate whether output has satisfied first criteria
        if "Counterexample" in nuxmv_output:
            print("Counterexample found for output threshold criteria")
        else:
            print("No counterexample found for output threshold criteria")
            last_newline = nuxmv_output[:len(nuxmv_output) - 2].rindex("\n")
            print(nuxmv_output[last_newline:])

            # Run nuXmv program 2
            result = subprocess.run(["../smt/nuxmv/nuxmv.exe", "../smt/nn_2.smv"], capture_output=True, text=True)
            nuxmv_output = result.stdout

            # Evaluate whether output has satisfied second criteria
            if "Counterexample" in nuxmv_output:
                print("Counterexample for output robustness criteria")
            else:
                print("No counterexample for output robustness criteria")
                last_newline = nuxmv_output[:len(nuxmv_output) - 2].rindex("\n")
                print(nuxmv_output[last_newline:])

                # Display the output for this proven result
                new_input_3 = str(int(parsed_smt['input_3']) + int(parsed_smt['input_3_difference']))
                new_input_color = [parsed_smt['input_1'], parsed_smt['input_2'], new_input_3]
                new_normalized_color = utils.generate_array_from_color(new_input_color)

                output = nn.eval(new_normalized_color)[0]
                output_weights = nn.get_output_weights()
                result = utils.NN_Results(output, new_normalized_color, output_weights, trainings_per_loop * count)
                result.print()

                break

        print()

        count += 1

        # Stop gap so the loop doesn't run out of control
        if count > 30:
            break


#
# Compute the smt statistics from here after 20 trainings - round to an int
#
# if output_20 < 100:
#	parsed_smt['hidden_1_output_weight'] = round(result.nn_ho_weights[0])
#	parsed_smt['hidden_2_output_weight'] = round(result.nn_ho_weights[1])
# parsed_smt['output_threshold'] = 100
# This will use the smt to prove that the result should have been less than 100 now, given the current weights

if __name__ == "__main__":
	main()