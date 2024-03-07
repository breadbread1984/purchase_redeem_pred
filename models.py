#!/usr/bin/python3

import tensorflow as tf

def Predictor(rnn_layer_num = 2, channel = 32):
  inputs = tf.keras.Input((None, 2)) # inputs.shape = (batch, seq_len, 2)
  rnn = tf.keras.layers.RNN([tf.keras.layers.GRUCell(channel) for i in range(rnn_layer_num)], return_sequence = True, return_state = True)
  outputs = rnn(inputs)
  results, state = outputs[0], outputs[1:] # results.shape = (batch, seq_len, channel)
  results = tf.keras.layers.Dense(2)(results) # results.shape = (batch, seq_len, 2)
  return tf.keras.Model(inputs = inputs, outputs = results)

