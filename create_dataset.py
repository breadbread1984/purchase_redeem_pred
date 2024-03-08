#!/usr/bin/python3

from absl import flags, app
from os.path import exists, join
from csv import DictReader
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_csv', default = None, help = 'path to user_balance_table.csv')
  flags.DEFINE_string('output', default = 'dataset.tfrecord', help = 'path to output npy')

def main(unused_argv):
  samples = dict()
  with open(FLAGS.input_csv, 'r') as f:
    csv = DictReader(f)
    for row in csv:
      if row['report_date'] not in samples: samples[row['report_date']] = (0,0)
      purchase, redeem = samples[row['report_date']]
      samples[row['report_date']] = (purchase + int(row['total_purchase_amt']), redeem + int(row['total_redeem_amt']))
  data = list()
  for key, value in samples.items():
    data.append((key, value[0], value[1]))
  data = list(sorted(data, key = lambda x: x[0]))
  sequence = np.stack([[d[1] for d in data],
                       [d[2] for d in data]], axis = -1) # sequence.shape = (seq_len, 2)
  writer = tf.io.TFRecordWriter(FLAGS.output)
  sample = tf.train.Example(features = tf.train.Features(
    feature = {
      'x': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(sequence[:-1]).numpy()])),
      'y': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(sequence[1:]).numpy()])),
      'len': tf.train.Feature(int64_list = tf.train.Int64List(value = [sequence.shape[0] - 1])),
      'min': tf.train.Feature(int64_list = tf.train.Int64List(value = [np.min(sequence[:,0]), np.min(sequence[:,1])])),
      'max': tf.train.Feature(int64_list = tf.train.Int64List(value = [np.max(sequence[:,0]), np.max(sequence[:,1])]))
    }
  ))
  writer.write(sample.SerializeToString())
  writer.close()
  print('purchase range: [%d, %d]' % (np.min(sequence[:,0]), np.max(sequence[:,0])))
  print('redeem range: [%d, %d]' % (np.min(sequence[:,1]), np.max(sequence[:,1])))
  print('date range: [%s, %s]' % (data[0][0], data[-1][0]))

if __name__ == "__main__":
  add_options()
  app.run(main)

