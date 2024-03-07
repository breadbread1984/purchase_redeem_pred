#!/usr/bin/python3

from absl import flags, app
from os.path import exists, join
from csv import DictReader
import pickle

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_csv', default = None, help = 'path to user_balance_table.csv')
  flags.DEFINE_string('output_pkl', default = 'dataset.pkl', help = 'path to output npy')

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
  with open(FLAGS.output_pkl, 'wb') as f:
    f.write(pickle.dumps(data))

if __name__ == "__main__":
  add_options()
  app.run(main)

