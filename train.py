#!/usr/bin/python3

from os import mkdir
from os.path import exists, join
from absl import flags, app
import tensorflow as tf
from models import Predictor

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate')
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
  model = Predictor(rnn_layer_num = FLAGS.layer_num, channel = FLAGS.channel)
  optimizer = tf.keras.optimizers.Adam(FLAGS.lr)
  dataset = tf.data.TFRecordDataset(FLAGS.dataset).map(parse_function).batch(1)

  if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
  checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer)
  checkpoint.restore(tf.train.latest_checkpoint(join(FLAGS.ckpt, 'ckpt')))

  for epoch in range(FLAGS.epoch):
    train_iter = iter(dataset)
    for x, y in train_iter:
      # purchase range: [8962232, 952479658]
      # redeem range: [1616635, 547295931]
      states = [tf.zeros((1, FLAGS.channel)) for i in range(FLAGS.layer_num)]
      with tf.GradientTape() as tape:
        pred, *latest_states = model([x, *states])
        loss = tf.keras.losses.MeanAbsoluteError()(y, pred)
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print('epoch %d MAE: %f' % (epoch, loss))
    checkpoint.save(join(FLAGS.ckpt, 'ckpt'))

if __name__ == "__main__":
  add_options()
  app.run(main)

