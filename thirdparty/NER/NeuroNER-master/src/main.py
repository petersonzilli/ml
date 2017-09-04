'''
CUDA_VISIBLE_DEVICES="" python3.5 main.py &
CUDA_VISIBLE_DEVICES=1 python3.5 main.py &
CUDA_VISIBLE_DEVICES=2 python3.5 main.py &
CUDA_VISIBLE_DEVICES=3 python3.5 main.py &
perl conlleval < 000050_train.txt > 000050_train_evaluation.txt
perl conlleval < 020000_test.txt > 020000_test_evaluation.txt
perl conlleval < 040000_test.txt > 040000_test_evaluation.txt
perl conlleval < 074930_test.txt > 074930_test_evaluation.txt
perl conlleval < 029972_test.txt > 029972_test_evaluation.txt
'''
from __future__ import print_function
import tensorflow as tf
import os
import collections
import utils
import networkx as nx
import numpy as np
import matplotlib
import copy
import subprocess
import utils_nlp
import re
from matplotlib.cbook import ls_mapper
import distutils
import distutils.util
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import token
import sklearn.preprocessing
import sklearn.metrics
import dataset as ds
import codecs
import time
import math
import random
import assess_model
import configparser
from pprint import pprint
from entity_lstm import EntityLSTM

print('TensorFlow version: {0}'.format(tf.__version__))




def load_token_embeddings(sess, W, dataset, parameters):
    # Load embeddings
    # https://github.com/dennybritz/cnn-text-classification-tf/issues/17
    print('Load embeddings')
    #full_word_embeddings_folder =os.path.join('..','data','word_vectors')
    #full_word_embeddings_filepath = os.path.join(full_word_embeddings_folder,'glove.6B.{0}d.txt'.format(token_embedding_size))
    file_input = codecs.open(parameters['token_pretrained_embedding_filepath'], 'r', 'UTF-8')
    count = -1
#     case_sensitive = False
#     initial_weights = np.random.uniform(-0.25,0.25,(vocabulary_size, token_embedding_size))
    initial_weights = sess.run(W.read_value())
    token_to_vector = {}
    for cur_line in file_input:
        count += 1
        #if count > 1000:break
        cur_line = cur_line.strip()
        cur_line = cur_line.split(' ')
        if len(cur_line)==0:continue
        token = cur_line[0]
        vector =cur_line[1:]
        token_to_vector[token] = vector

    number_of_loaded_word_vectors = 0
    number_of_token_original_case_found = 0
    number_of_token_lowercase_found = 0
    number_of_token_lowercase_normalized_found = 0
    for token in dataset.token_to_index.keys():
        # TODO: shouldn't it apply to token_to_index instead?
#         if not case_sensitive: token = token.lower()
        # For python 2.7
#         if token not in dataset.token_to_index.viewkeys():continue
        # For python 3.5
        if token in token_to_vector.keys():
            initial_weights[dataset.token_to_index[token]] = token_to_vector[token]
            number_of_token_original_case_found += 1
        elif token.lower() in token_to_vector.keys():
            initial_weights[dataset.token_to_index[token]] = token_to_vector[token.lower()]
            number_of_token_lowercase_found += 1
        elif re.sub('\d', '0', token.lower()) in token_to_vector.keys():
            initial_weights[dataset.token_to_index[token]] = token_to_vector[re.sub('\d', '0', token.lower())]
            number_of_token_lowercase_normalized_found += 1
        else:
            continue
        number_of_loaded_word_vectors += 1
    file_input.close()
    print("number_of_token_original_case_found: {0}".format(number_of_token_original_case_found))
    print("number_of_token_lowercase_found: {0}".format(number_of_token_lowercase_found))
    print("number_of_token_lowercase_normalized_found: {0}".format(number_of_token_lowercase_normalized_found))
    print('number_of_loaded_word_vectors: {0}'.format(number_of_loaded_word_vectors))
    print("len(dataset.token_to_index): {0}".format(len(dataset.token_to_index)))
    print("len(dataset.index_to_token): {0}".format(len(dataset.index_to_token)))
#     sess.run(tf.global_variables_initializer())
    sess.run(W.assign(initial_weights))
    print('Load embeddings completed')


def train_step(sess, dataset, sequence_number, train_op, global_step, model, transition_params_trained, parameters):

    '''

    '''
    # Perform one iteration
    '''
    x_batch = range (20)
    y_batch = [[0,0,0,1,0]] * 20
    print('y_batch: {0}'.format(y_batch))
    feed_dict = {
      input_token_indices: x_batch,
      input_label_indices_vector: y_batch
    }
    '''

    token_indices_sequence = dataset.token_indices['train'][sequence_number]
    for i, token_index in enumerate(token_indices_sequence):
        if token_index in dataset.infrequent_token_indices and np.random.uniform() < 0.5:
#                         print("token_indices_sequence[i]: {0}".format(token_indices_sequence[i]))
            token_indices_sequence[i] = dataset.token_to_index[dataset.UNK]
#                         print("changed to UNK: {0}".format(token_indices_sequence[i]))

#     label_indices_sequence = dataset.label_indices['train'][sequence_number]
#     label_vector_indices_sequence = dataset.label_vector_indices['train'][sequence_number]
#     character_indices_padded_sequence = dataset.character_indices_padded['train'][sequence_number]
#     token_lengths_sequence = dataset.token_lengths['train'][sequence_number]

    #print('len(token_indices_sequence): {0}'.format(len(token_indices_sequence)))
    # TODO: match the names
    if len(token_indices_sequence)<2: return transition_params_trained
    feed_dict = {
      model.input_token_indices: token_indices_sequence,
      model.input_label_indices_vector: dataset.label_vector_indices['train'][sequence_number],
      model.input_token_indices_character: dataset.character_indices_padded['train'][sequence_number],
      model.input_token_lengths: dataset.token_lengths['train'][sequence_number],
      model.input_label_indices_flat: dataset.label_indices['train'][sequence_number],
#                   model.input_crf_transition_params: transition_params_random,
      model.dropout_keep_prob: parameters['dropout_rate']
    }
    _, _, loss, accuracy, transition_params_trained = sess.run(
                    [train_op, global_step, model.loss, model.accuracy, model.transition_params],
                    feed_dict)

    #print('loss: {0:0.3f}\taccuracy: {1:0.3f}'.format(loss, accuracy))
    #print('predictions: {0}'.format(predictions))
    return transition_params_trained

def evaluate_model(sess, dataset, dataset_type, model, transition_params_trained, step, stats_graph_folder, epoch_number, parameters):
    '''

    '''

    print('evaluate_model')
    all_predictions = []
    all_y_true = []
    output_filepath = os.path.join(stats_graph_folder, '{2:03d}_{1:06d}_{0}.txt'.format(dataset_type,step,epoch_number))
    output_file = open(output_filepath, 'w')

    # TODO: merge with feed_dict?
    token_indices=dataset.token_indices[dataset_type]
    label_indices=dataset.label_indices[dataset_type]
    label_vector_indices=dataset.label_vector_indices[dataset_type]
    character_indices_padded = dataset.character_indices_padded[dataset_type]
    token_lengths = dataset.token_lengths[dataset_type]
    #for i in range( 200 ):
    for i in range(len(token_indices)):

#         token_indices_sequence = token_indices[i]
#         label_vector_indices_sequence = label_vector_indices[i]
#         character_indices_padded_sequence = character_indices_padded[i]
#         token_lengths_sequence = token_lengths[i]
        #print('label_vector_indices_sequence:\n{0}'.format(label_vector_indices_sequence))

        feed_dict = {
          model.input_token_indices: token_indices[i],
          model.input_token_indices_character: character_indices_padded[i],
          model.input_token_lengths: token_lengths[i],
          model.input_label_indices_vector: label_vector_indices[i],
#                       model.input_crf_transition_params: transition_params_trained,
          model.dropout_keep_prob: 1.
        }
        #print('type(input_token_indices): {0}'.format(type(input_token_indices)))
        #print('type(input_label_indices_vector): {0}'.format(type(input_label_indices_vector)))
        #print('type(feed_dict): {0}'.format(type(feed_dict)))
        '''
        predictions = sess.run(
                        predictions_,
                        feed_dict=feed_dict)
        predictions = predictions.tolist()
        '''
        unary_scores, predictions = sess.run([model.unary_scores, model.predictions], feed_dict)
        if parameters['use_crf']:
            predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params_trained)
        else:
            predictions = predictions.tolist()

        output_string = ''
        for prediction, token,gold_label in zip(predictions,dataset.tokens[dataset_type][i],dataset.labels[dataset_type][i]):
            output_string += '{0} {1} {2}\n'.format(token, gold_label, dataset.index_to_label[prediction])
        output_file.write(output_string+'\n')

        #print('predictions: {0}'.format(predictions)
        all_predictions.extend(predictions)
        all_y_true.extend(label_indices[i])

    output_file.close()


    #print('all_predictions: {0}'.format(all_predictions))
    #print('all_y_true: {0}'.format(all_y_true))
    # TODO: make pretty
    print('dataset.unique_labels: {0}'.format(dataset.unique_labels))
    remove_b_and_i_from_predictions(all_predictions, dataset)
    remove_b_and_i_from_predictions(all_y_true, dataset)
    #print(sklearn.metrics.classification_report(all_y_true, all_predictions,digits=4,labels=range(len(dataset.unique_labels)-1), target_names=dataset.unique_labels[:-1]))
    print(sklearn.metrics.classification_report(all_y_true, all_predictions,digits=4,labels=dataset.unique_label_indices_of_interest,
                                                                                                  target_names=dataset.unique_labels_of_interest))
    print(sklearn.metrics.classification_report(all_y_true, all_predictions,digits=4,labels=range(len(dataset.unique_labels)), target_names=dataset.unique_labels))

    return all_predictions, all_y_true, output_filepath



def main():


    #### Parameters - start
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(os.path.join('.','parameters.ini'))
    nested_parameters = utils.convert_configparser_to_dictionary(conf_parameters)
    parameters = {}
    for k,v in nested_parameters.items():
        parameters.update(v)
    for k,v in parameters.items():
        if k in ['remove_unknown_tokens','character_embedding_dimension','character_lstm_hidden_state_dimension','token_embedding_dimension','token_lstm_hidden_state_dimension',
                 'patience','maximum_number_of_epochs','maximum_training_time','number_of_cpu_threads','number_of_gpus']:
            parameters[k] = int(v)
        if k in ['dropout_rate']:
            parameters[k] = float(v)
        if k in ['use_character_lstm','is_character_lstm_bidirect','is_token_lstm_bidirect','use_crf']:
            parameters[k] = distutils.util.strtobool(v)
    pprint(parameters)

    # Load dataset
    dataset_filepaths = {}
    dataset_filepaths['train'] = os.path.join(parameters['dataset_text_folder'], 'train.txt')
    dataset_filepaths['valid'] = os.path.join(parameters['dataset_text_folder'], 'valid.txt')
    dataset_filepaths['test']  = os.path.join(parameters['dataset_text_folder'], 'test.txt')
    dataset = ds.Dataset()
    dataset.load_dataset(dataset_filepaths, parameters)


    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          device_count={'CPU': 1, 'GPU': 1},
          allow_soft_placement=True, #  automatically choose an existing and supported device to run the operations in case the specified one doesn't exist
          log_device_placement=False
          )

        sess = tf.Session(config=session_conf)

        with sess.as_default():
            model = EntityLSTM(dataset, parameters)

            # Define training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            if parameters['optimizer'] == 'adam':
                optimizer = tf.train.AdamOptimizer(1e-3)
            elif parameters['optimizer'] == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(0.005)
            else:
                raise ValueError("The lr_method parameter must be either adam or sgd.")

            # https://github.com/google/prettytensor/issues/6
            # https://www.tensorflow.org/api_docs/python/framework/graph_collections

            #print('tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) : {0}'.format(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) ))
            #print('tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) : {0}'.format(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) ))
            #print('tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) : {0}'.format(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) ))

            # https://github.com/blei-lab/edward/issues/286#ref-pullrequest-181330211 : utility function to get all tensorflow variables a node depends on


            grads_and_vars = optimizer.compute_gradients(model.loss)

            # By defining a global_step variable and passing it to the optimizer we allow TensorFlow handle the counting of training steps for us.
            # The global step will be automatically incremented by one every time you execute train_op.
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Load pretrained token embeddings
            if not parameters['token_pretrained_embedding_filepath'] == '':
                load_token_embeddings(sess, model.W, dataset, parameters)


            estop = False  # early stop
            start_time = time.time()
            experiment_timestamp = utils.get_current_time_in_miliseconds()
            results = {}
            #results['model_options'] = copy.copy(model_options)
            #results['model_options'].pop('optimizer', None)
            results['epoch'] = {}
            # save/initialize execution details
            results['execution_details'] = {}
            results['execution_details']['train_start'] = start_time
            results['execution_details']['time_stamp'] = experiment_timestamp
            results['execution_details']['early_stop'] = False
            results['execution_details']['keyboard_interrupt'] = False
            results['execution_details']['num_epochs'] = 0
            results['model_options'] = copy.copy(parameters)

            dataset_name = utils.get_basename_without_extension(parameters['dataset_text_folder']) #opts.train.replace('/', '_').split('.')[0] # 'conll2003en'
            model_name = '{0}_{1}'.format(dataset_name, results['execution_details']['time_stamp'])

            output_folder=os.path.join('..', 'output')
            stats_graph_folder=os.path.join(output_folder, model_name) # Folder where to save graphs
            utils.create_folder_if_not_exists(output_folder)
            print('stats_graph_folder: {0}'.format(stats_graph_folder))
            utils.create_folder_if_not_exists(stats_graph_folder)
            model_folder = os.path.join(stats_graph_folder, 'model')
            utils.create_folder_if_not_exists(model_folder)

            step = 0
            bad_counter = 0
            previous_best_valid_f1_score = 0
            transition_params_trained = np.random.rand(len(dataset.unique_labels),len(dataset.unique_labels))
            try:
                while True:
                    epoch_number = math.floor(step / len(dataset.token_indices['train']))
                    print('epoch_number: {0}'.format(epoch_number))

                    epoch_start_time = time.time()

                    #print('step: {0}'.format(step))

                    # Train model: loop over all sequences of training set with shuffling
                    sequence_numbers=list(range(len(dataset.token_indices['train'])))
                    random.shuffle(sequence_numbers)
                    for sequence_number in sequence_numbers:
                        transition_params_trained = train_step(sess, dataset, sequence_number, train_op, global_step, model, transition_params_trained, parameters)
                        step += 1
                        if sequence_number % 100 == 0:
                            print('.',end='', flush=True)
                            #break

                    # Evaluate model
                    print('step: {0}'.format(step))
                    all_predictions = {}
                    all_y_true  = {}
                    output_filepaths = {}
                    for dataset_type in ['train', 'valid', 'test']:
                        print('dataset_type:     {0}'.format(dataset_type))
                        all_predictions[dataset_type], all_y_true[dataset_type], output_filepaths[dataset_type] = evaluate_model(sess, dataset, dataset_type, model, transition_params_trained, step, stats_graph_folder, epoch_number, parameters)
                        model_options = None

                    # Save and plot results
                    # TODO: remove uidx
                    uidx = 0
                    results['epoch'][epoch_number] = []
                    results['execution_details']['num_epochs'] = epoch_number

                    epoch_elapsed_training_time = time.time() - epoch_start_time
                    print('epoch_elapsed_training_time: {0:02f} seconds'.format(epoch_elapsed_training_time))

                    assess_model.assess_and_save(results, dataset, model_options, all_predictions, all_y_true, stats_graph_folder, epoch_number, uidx, epoch_start_time)
                    assess_model.plot_f1_vs_epoch(results, stats_graph_folder, 'f1_score')
                    assess_model.plot_f1_vs_epoch(results, stats_graph_folder, 'accuracy_score')

                    # CoNLL evaluation script
                    for dataset_type in ['train', 'valid', 'test']:
                        conll_evaluation_script = os.path.join('.', 'conlleval')
                        conll_output_filepath = '{0}_conll_evaluation.txt'.format(output_filepaths[dataset_type])
                        shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script, output_filepaths[dataset_type], conll_output_filepath)
                        print('shell_command: {0}'.format(shell_command))
                        #subprocess.call([shell_command])
                        os.system(shell_command)
                        conll_parsed_output = utils_nlp.get_parsed_conll_output(conll_output_filepath)
                        print('conll_parsed_output: {0}'.format(conll_parsed_output))
                        results['epoch'][epoch_number][0][dataset_type]['conll'] = conll_parsed_output
                        results['epoch'][epoch_number][0][dataset_type]['f1_conll'] = {}
                        results['epoch'][epoch_number][0][dataset_type]['f1_conll']['micro'] = results['epoch'][epoch_number][0][dataset_type]['conll']['all']['f1']
                    assess_model.plot_f1_vs_epoch(results, stats_graph_folder, 'f1_conll', from_json=False)

                    #end_time = time.time()
                    #results['execution_details']['train_duration'] = end_time - start_time
                    #results['execution_details']['train_end'] = end_time

                    # Early stop
                    valid_f1_score = results['epoch'][epoch_number][0]['valid']['f1_score']['micro']
                    if  valid_f1_score > previous_best_valid_f1_score:
                        bad_counter = 0
                        previous_best_valid_f1_score = valid_f1_score
                    else:
                        bad_counter += 1


                    if bad_counter > parameters['patience']:
                        print('Early Stop!')
                        results['execution_details']['early_stop'] = True
                        break

                    if epoch_number > parameters['maximum_number_of_epochs']: break

            except KeyboardInterrupt:
                results['execution_details']['keyboard_interrupt'] = True
        #         assess_model.save_results(results, stats_graph_folder)
                print('Training interrupted')

            print('Finishing the experiment')
            end_time = time.time()
            results['execution_details']['train_duration'] = end_time - start_time
            results['execution_details']['train_end'] = end_time
            assess_model.save_results(results, stats_graph_folder)

    sess.close() # release the session's resources


def remove_b_and_i_from_predictions(predictions, dataset):
    '''

    '''
    for prediction_number, prediction  in enumerate(predictions):
        prediction = int(prediction)
        prediction_label = dataset.index_to_label[prediction]
        #print('prediction_label : {0}'.format(prediction_label ))
        if prediction_label.startswith('I-'):
            new_prediction_label = 'B-' + prediction_label[2:]
            #print('new_prediction_label: {0}'.format(new_prediction_label))
            if new_prediction_label in dataset.unique_labels:
                predictions[prediction_number] = dataset.label_to_index[new_prediction_label]
                #print(prediction)


if __name__ == "__main__":
    while True:
        main()
    #cProfile.run('main()') # if you want to do some profiling


