#!/usr/bin/python3

from absl import flags, app
import tensorflow as tf

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'dataset.tfrecord', help = 'path to tfrecord')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to ckpt directory')
  flags.DFEINE_integer('channel', default = 32, help = 'channel')
  flags.DEFINE_integer('layer_num', default = 2, help = 'layer number')
  flags.DEFINE_string('output_csv', default = 'output.csv', help = 'path to output csv')

def parse_function(serialized_example):
  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'x': tf.io.FixedLenFeature((), dtype = tf.string),
      'y': tf.io.FixedLenFeature((), dtype = tf.string),
      'len': tf.io.FixedLenFeature((), dtype = tf.int64),
      'min': tf.io.FixedLenFeature((2,), dtype = tf.int64),
      'max': tf.io.FixedLenFeature((2,), dtype = tf.int64),
    }
  )
  x = tf.io.parse_tensor(feature['x'], out_type = tf.int64)
  y = tf.io.parse_tensor(feature['y'], out_type = tf.int64)
  length = tf.cast(feature['len'], dtype = tf.int32)
  x = tf.cast(tf.reshape(x, (length, 2)), dtype = tf.float32)
  y = tf.cast(tf.reshape(y, (length, 2)), dtype = tf.float32)
  min_value = tf.cast(feature['min'], dtype = tf.float32)
  min_value = tf.reshape(min_value, (1, 2))
  max_value = tf.cast(feature['max'], dtype = tf.float32)
  max_value = tf.reshape(max_value, (1, 2))
  x = (x - min_value) / (max_value - min_value)
  y = (y - min_value) / (max_value - min_value)
  return x, y

def main(unused_argv):
  dataset = tf.data.TFRecordDataset(FLAGS.dataset).map(parse_function).batch(1)
  states = [tf.zeros((1, FLAGS.channel)) for i in range(FLAGS.layer_num)]
  for x, y in dataset
    # NOTE: x.shape = (1, seq_len - 1, 2) # [t0,tn-1]
    # NOTE: y.shape = (1, seq_len - 1, 2) # [t1,tn]
    preds, *states = model([x, *states]) # pred.shape = (1, seq_len, 2)
    break
  inputs = y[:,-1:,:] # inputs.shape = (1, 1, 2)
  outputs = tf.zeros((1,0,2)) # outputs.shape = (1, 0, 2)
  for i in range(30):
    preds, *states = model([inputs, *states]) # pred.shape = (1,1,2)
    inputs = preds
    outputs = tf.concate([outputs, preds], axis = 1) # outputs.shape = (1, seq_len+1, 2)
  outputs = outputs.numpy()
  scale = np.array([952479658. - 8962232., 547295931. - 1616635.], dtype = np.float32).reshape((1,1,2))
  min_values = np.array([8962232., 1616635.], dtype = np.float32).reshape((1,1,2))
  outputs = outputs * scale + min_values
  outputs = outputs[0].astype(np.int32)
  with open(FLAGS.output, 'w') as f:
    for i, output in enumerate(outputs):
      date = 20140901 + i
      f.write('%d,%d,%d' & date, output[0], output[1])

if __name__ == "__main__":
  add_options()
  app.run(main)
