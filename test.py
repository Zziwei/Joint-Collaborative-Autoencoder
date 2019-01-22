"""
Ziwei Zhu
Computer Science and Engineering Department, Texas A&M University
zhuziwei@tamu.edu
"""
from data_preprocessor import *
import tensorflow as tf
import time
import argparse
import os
from JCA import JCA

if __name__ == '__main__':
    neg_sample_rate = 1

    date = time.strftime('%y-%m-%d', time.localtime())
    current_time = time.strftime('%H:%M:%S', time.localtime())
    data_name = 'ml-1m'
    base = 'u'

    parser = argparse.ArgumentParser(description='JCA')

    parser.add_argument('--train_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1500)
    parser.add_argument('--display_step', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--lambda_value', type=float, default=0.001) 
    parser.add_argument('--margin', type=float, default=0.15) 
    parser.add_argument('--optimizer_method', choices=['Adam', 'Adadelta', 'Adagrad', 'RMSProp', 'GradientDescent',
                                                       'Momentum'], default='Adam')
    parser.add_argument('--g_act', choices=['Sigmoid', 'Relu', 'Elu', 'Tanh', "Identity"], default='Sigmoid')
    parser.add_argument('--f_act', choices=['Sigmoid', 'Relu', 'Elu', 'Tanh', "Identity"], default='Sigmoid')
    parser.add_argument('--U_hidden_neuron', type=int, default=160)
    parser.add_argument('--I_hidden_neuron', type=int, default=160)
    parser.add_argument('--base', type=str, default=base)
    parser.add_argument('--neg_sample_rate', type=int, default=neg_sample_rate)
    args = parser.parse_args()

    sess = tf.Session()

    train_R, test_R = yelp.test()
    metric_path = './metric_results_test/' + date + '/'
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
    metric_path = metric_path + '/' + str(parser.description) + "_" + str(current_time)
    jca = JCA(sess, args, train_R, test_R, metric_path, date, data_name)
    jca.run(train_R, test_R)
