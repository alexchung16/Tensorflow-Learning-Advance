#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : cosine_decay_with_warmup.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/7/31 下午4:38
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf

summary_path = './summary'
method = 'cosine_decay_with_warmup'

max_step = 20000
total_decay_step = 15000 # under normal conditions the total_step equal to max_step
base_learning_rate = 0.01
warmup_learning_rate = 0.0001
warmup_steps = 2000
hold_base_rate_steps = 1000
alpha = 0.00001

summary_step = 10

def main():

    global_step_op = tf.train.get_or_create_global_step()

    learning_rate = cosine_decay_with_warmup(learning_rate_base=base_learning_rate,
                                             total_decay_steps=total_decay_step,
                                             alpha = alpha,
                                             warmup_learning_rate=warmup_learning_rate,
                                             warmup_steps=warmup_steps,
                                             hold_base_rate_steps=hold_base_rate_steps,
                                             global_step=global_step_op)
    tf.summary.scalar("cosine_decay_with_warmup", learning_rate)
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


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_decay_steps,
                             alpha = 0.0,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
  """Cosine decay schedule with warm up period.

  Cosine annealing learning rate as described in:
    Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
    ICLR 2017. https://arxiv.org/abs/1608.03983
  In this schedule, the learning rate grows linearly from warmup_learning_rate
  to learning_rate_base for warmup_steps, then transitions to a cosine decay
  schedule.

  Args:
    global_step: int64 (scalar) tensor representing global step.
    learning_rate_base: base learning rate.
    total_decay_steps: total number of learning rate decay steps.
    alpha: Minimum learning rate value as a fraction of learning_rate.
    warmup_learning_rate: initial learning rate for warm up.
    warmup_steps: number of warmup steps.
    hold_base_rate_steps: Optional number of steps to hold base learning rate
      before decaying.

  Returns:
    If executing eagerly:
      returns a no-arg callable that outputs the (scalar)
      float tensor learning rate given the current value of global_step.
    If in a graph:
      immediately returns a (scalar) float tensor representing learning rate.

  Raises:
    ValueError: if warmup_learning_rate is larger than learning_rate_base,
      or if warmup_steps is larger than total_steps.
  """
  if total_decay_steps < warmup_steps:
    raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')
  def eager_decay_rate():
    """Callable to compute the learning rate."""
    # reference cosine_decay
    # cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
    # decayed = (1 - alpha) * cosine_decay + alpha
    # decayed_learning_rate = learning_rate * decayed
    # where alpha = 0
    # global_step = global_step - (warmup_steps + hold_base_rate_steps)
    # decay_step = total_steps - (warmup_steps + hold_base_rate_steps)
    # learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
    #     np.pi *(tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
    #     ) / float(total_decay_steps - warmup_steps - hold_base_rate_steps)))
    learning_rate = tf.train.cosine_decay(learning_rate=learning_rate_base,
                                          decay_steps=total_decay_steps - warmup_steps - hold_base_rate_steps,
                                          global_step= global_step - warmup_steps - hold_base_rate_steps,
                                          alpha=alpha)
    if hold_base_rate_steps > 0:
      learning_rate = tf.where(
          global_step > warmup_steps + hold_base_rate_steps,
          learning_rate, learning_rate_base)
    if warmup_steps > 0:
      if learning_rate_base < warmup_learning_rate:
        raise ValueError('learning_rate_base must be larger or equal to '
                         'warmup_learning_rate.')
      slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
      warmup_rate = slope * tf.cast(global_step,
                                    tf.float32) + warmup_learning_rate
      learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                               learning_rate)
    return tf.where(global_step > total_decay_steps, alpha, learning_rate,
                    name='learning_rate')

  if tf.executing_eagerly():
    return eager_decay_rate
  else:
    return eager_decay_rate()


if __name__ == "__main__":
    main()