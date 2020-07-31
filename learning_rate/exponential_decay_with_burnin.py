#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : exponential_decay_with_burnin.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/7/31 下午4:03
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import tensorflow as tf

summary_path = './summary'
method = 'exponential_decay_with_burnin'

max_step = 20000
base_learning_rate = 0.01
decay_rate = 0.98
decay_steps = 2000
burnin_learning_rate = 0.001  # burn-in learning rate
burnin_steps = 2000


summary_step = 10

def main():

    global_step_op = tf.train.get_or_create_global_step()

    learning_rate = exponential_decay_with_burnin(learning_rate_base=base_learning_rate,
                                                  learning_rate_decay_steps=decay_steps,
                                                  learning_rate_decay_factor=decay_rate,
                                                  burnin_learning_rate=burnin_learning_rate,
                                                  burnin_steps=burnin_steps,
                                                  staircase=True,
                                                  global_step=global_step_op)
    tf.summary.scalar("exponential_decay_with_burnin", learning_rate)
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


def exponential_decay_with_burnin(global_step,
                                  learning_rate_base,
                                  learning_rate_decay_steps,
                                  learning_rate_decay_factor,
                                  burnin_learning_rate=0.0,
                                  burnin_steps=0,
                                  min_learning_rate=0.0,
                                  staircase=True):
  """Exponential decay schedule with burn-in period.

  In this schedule, learning rate is fixed at burnin_learning_rate
  for a fixed period, before transitioning to a regular exponential
  decay schedule.

  Args:
    global_step: int tensor representing global step.
    learning_rate_base: base learning rate.
    learning_rate_decay_steps: steps to take between decaying the learning rate.
      Note that this includes the number of burn-in steps.
    learning_rate_decay_factor: multiplicative factor by which to decay
      learning rate.
    burnin_learning_rate: initial learning rate during burn-in period.  If
      0.0 (which is the default), then the burn-in learning rate is simply
      set to learning_rate_base.
    burnin_steps: number of steps to use burnin learning rate.
    min_learning_rate: the minimum learning rate.
    staircase: whether use staircase decay.

  Returns:
    If executing eagerly:
      returns a no-arg callable that outputs the (scalar)
      float tensor learning rate given the current value of global_step.
    If in a graph:
      immediately returns a (scalar) float tensor representing learning rate.
  """
  if burnin_learning_rate == 0:
    burnin_learning_rate = learning_rate_base

  def eager_decay_rate():
    """Callable to compute the learning rate."""
    post_burnin_learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step - burnin_steps,
        learning_rate_decay_steps,
        learning_rate_decay_factor,
        staircase=staircase)
    if callable(post_burnin_learning_rate):
      post_burnin_learning_rate = post_burnin_learning_rate()
    return tf.maximum(tf.where(
        tf.less(tf.cast(global_step, tf.int32), tf.constant(burnin_steps)),
        tf.constant(burnin_learning_rate),
        post_burnin_learning_rate), min_learning_rate, name='learning_rate')

  if tf.executing_eagerly():
    return eager_decay_rate
  else:
    return eager_decay_rate()


if __name__ == "__main__":
    main()