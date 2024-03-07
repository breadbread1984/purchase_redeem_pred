#!/usr/bin/python3

from os import mkdir
from os.path import exists, join
from absl import flags, app
import tensorflow as tf
from models import Predictor

FLAGS = flags.FLAGS

def add_options():
  flags.DFEINE_string('lr', default = 1e-3, help = 'learning rate')
  flags.DEFINE_string('dataset', default = 'dataset.tfrecord', help = 'path to tfrecord')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'paht to checkpoint')
  flags.DEFINE_integer('epoch', default = 5, help = 'epoch')
  flags.DEFINE_integer('channel', default = 32, help = 'channel')
  flags.DEFINE_integer('layer_num', default = 2, help = 'layer num')

def parse_function(serialized_example):
  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'x': tf.io.FixedLenFeature((), dtype = tf.string),
      'y': tf.io.FixedLenFeature((), dtype = tf.string),
      'len': tf.io.FixedLenFeature((), dtype = tf.int32)
    }
  )
  x = tf.io.parse_tensor(feature['x'], out_type = tf.int64)
  y = tf.io.parse_tensor(feature['y'], out_type = tf.int64)
  length = tf.cast(feature['len'], dtype = tf.int32)
  x = tf.reshape(x, (length, 2))
  y = tf.reshape(y, (length, 2))
  return tf.cast(x, dtype = tf.int32), tf.cast(y, dtype = tf.int32)

def main(unused_argv):
  model = Predictor(rnn_layer_num = FLAGS.layer_num, channel = FLAGS.channel)
  optimizer = tf.keras.optimizers.Adam(FLAGS.lr)
  dataset = tf.data.TFRecordDataset(FLAGS.dataset).map(parse_function).batch(1)

  if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
  checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer)
  checkpoint.restore(tf.train.latest_checkpoint(join(FLAGS.ckpt, 'ckpt')))

  for epoch in range(FLAGS.epoch):
    train_iter = iter(dataset)
    for x, y in train_iter:
      x = tf.cast(x, dtype = tf.float32)
      y = tf.cast(y, dtype = tf.float32)
      
