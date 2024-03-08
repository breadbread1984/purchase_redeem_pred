#!/usr/bin/python3

import tensorflow as tf

def Predictor(rnn_layer_num = 2, channel = 32):
  inputs = tf.keras.Input((None, 2)) # inputs.shape = (batch, seq_len, 2)
  results = tf.keras.layers.LayerNormalization()(inputs)
  results = tf.keras.layers.Dense(channel, activation = tf.keras.activations.gelu)(results)
  results = tf.keras.layers.Dropout(0.1)(results)
  states = [tf.keras.Input((channel,)) for i in range(rnn_layer_num)] # state.shape = (batch, channel)
  rnn = tf.keras.layers.RNN([tf.keras.layers.GRUCell(channel) for i in range(rnn_layer_num)], return_sequences = True, return_state = True)
  outputs = rnn(results, initial_state = states)
  results, latest_states = outputs[0], outputs[1:] # results.shape = (batch, seq_len, channel)
  results = tf.keras.layers.LayerNormalization()(results)
  results = tf.keras.layers.Dense(2)(results)
  return tf.keras.Model(inputs = [inputs, *states], outputs = [results, *latest_states])

