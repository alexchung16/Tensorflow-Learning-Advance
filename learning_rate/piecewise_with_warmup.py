#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : piecewise_with_warmup.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/1 下午2:16
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import tensorflow as tf

summary_path = './summary'
method = 'piecewise_with_warmup'

max_step = 20000
base_learning_rate = 0.01
warmup_rates = 0.00001
decay_boundaries = [5000, 8000]
learning_rates = [base_learning_rate, base_learning_rate/10., base_learning_rate/100.]

warmup_steps = 2000
decay_boundaries_warmup = [warmup_steps, 5000, 8000]
learning_rates_warmup = [warmup_rates, base_learning_rate, base_learning_rate/10., base_learning_rate/100.]


summary_step = 10

def main():

    global_step_op = tf.train.get_or_create_global_step()

    learning_rate_no_warmup = manual_stepping(boundaries=decay_boundaries,
                                              rates = learning_rates,
                                              warmup=False,
                                              global_step=global_step_op,)
    tf.summary.scalar("piecewise_no_warmup", learning_rate_no_warmup)
    learning_rate_with_warmup = manual_stepping(boundaries=decay_boundaries_warmup,
                                              rates=learning_rates_warmup,
                                              warmup=True,
                                              global_step=global_step_op, )
    tf.summary.scalar("piecewise_with_warmup", learning_rate_with_warmup)
    summary_op = tf.summary.merge_all()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        summary_write = tf.summary.FileWriter(os.path.join(summary_path, method))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            if not coord.should_stop():
                for step in range(max_step):
                    if step % summary_step == 0:
                        summary, global_step = sess.run([summary_op, global_step_op], feed_dict={global_step_op:step})
                        summary_write.add_summary(summary, global_step=global_step)
                        summary_write.flush()

                    summary, global_step = sess.run([summary_op, global_step_op], feed_dict={global_step_op:step})

        except Exception as e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
            print('all threads are asked to stop!')


def manual_stepping(global_step, boundaries, rates, warmup=False):
  """Manually stepped learning rate schedule.

  This function provides fine grained control over learning rates.  One must
  specify a sequence of learning rates as well as a set of integer steps
  at which the current learning rate must transition to the next.  For example,
  if boundaries = [5, 10] and rates = [.1, .01, .001], then the learning
  rate returned by this function is .1 for global_step=0,...,4, .01 for
  global_step=5...9, and .001 for global_step=10 and onward.

  Args:
    global_step: int64 (scalar) tensor representing global step.
    boundaries: a list of global steps at which to switch learning
      rates.  This list is assumed to consist of increasing positive integers.
    rates: a list of (float) learning rates corresponding to intervals between
      the boundaries.  The length of this list must be exactly
      len(boundaries) + 1.
    warmup: Whether to linearly interpolate learning rate for steps in
      [0, boundaries[0]].

  Returns:
    If executing eagerly:
      returns a no-arg callable that outputs the (scalar)
      float tensor learning rate given the current value of global_step.
    If in a graph:
      immediately returns a (scalar) float tensor representing learning rate.
  Raises:
    ValueError: if one of the following checks fails:
      1. boundaries is a strictly increasing list of positive integers
      2. len(rates) == len(boundaries) + 1
      3. boundaries[0] != 0
  """
  if any([b < 0 for b in boundaries]) or any(
      [not isinstance(b, int) for b in boundaries]):
    raise ValueError('boundaries must be a list of positive integers')
  if any([bnext <= b for bnext, b in zip(boundaries[1:], boundaries[:-1])]):
    raise ValueError('Entries in boundaries must be strictly increasing.')
  if any([not isinstance(r, float) for r in rates]):
    raise ValueError('Learning rates must be floats')
  if len(rates) != len(boundaries) + 1:
    raise ValueError('Number of provided learning rates must exceed '
                     'number of boundary points by exactly 1.')

  if boundaries and boundaries[0] == 0:
    raise ValueError('First step cannot be zero.')

  if warmup and boundaries:
    slope = (rates[1] - rates[0]) * 1.0 / boundaries[0]
    warmup_steps = list(range(boundaries[0]))
    warmup_rates = [rates[0] + slope * step for step in warmup_steps]
    boundaries = warmup_steps + boundaries
    rates = warmup_rates + rates[1:]
  else:
    boundaries = [0] + boundaries
  num_boundaries = len(boundaries)

  def eager_decay_rate():
    """Callable to compute the learning rate."""
    rate_index = tf.reduce_max(tf.where(
        tf.greater_equal(global_step, boundaries),
        list(range(num_boundaries)),
        [0] * num_boundaries))
    return tf.reduce_sum(rates * tf.one_hot(rate_index, depth=num_boundaries),
                         name='learning_rate')
  if tf.executing_eagerly():
    return eager_decay_rate
  else:
    return eager_decay_rate()


if __name__ == "__main__":
    main()