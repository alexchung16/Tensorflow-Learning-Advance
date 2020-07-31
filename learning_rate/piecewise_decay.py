#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : piecewise_learning_rate.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/7/31 上午8:20
# @ Software   : PyCharm
#-------------------------------------------------------

# https://github.com/tensorflow/models/blob/v1.13.0/official/resnet/resnet_run_loop.py#L225
# https://github.com/tensorflow/models/blob/master/research/object_detection/utils/learning_schedules.py

import os
import tensorflow as tf

summary_path = './summary'
method = 'piecewise_learning_rate'

max_step = 20000
base_learning_rate = 0.001
decay_boundaries = [5000, 8000]
learning_rate_value = [base_learning_rate, base_learning_rate/10., base_learning_rate/100.]
summary_step = 10

def main():

    global_step_op = tf.train.get_or_create_global_step()

    # x = tf.get_variable(shape=[1], initializer=tf.random_normal_initializer(), name="x")
    learning_rate =  tf.train.piecewise_constant_decay(global_step_op,
                                                       boundaries=decay_boundaries,
                                                       values=learning_rate_value)
    tf.summary.scalar("learning_rate", learning_rate)
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

if __name__ == "__main__":
    main()










