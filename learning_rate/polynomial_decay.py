#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : polynomial_decay.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/7/31 下午3:29
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import tensorflow as tf

summary_path = './summary'
method = 'polynomial_decay'

max_step = 20000
base_learning_rate = 0.01
decay_steps = 10000
end_learning_rate = 0.00001
summary_step = 10

def main():

    global_step_op = tf.train.get_or_create_global_step()

    # global_step = min(global_step, decay_steps)
    # decayed_learning_rate = (learning_rate - end_learning_rate) *
    #                           (1 - global_step / decay_steps) ^ (power) + end_learning_rate
    learning_rate_power_1 =  tf.train.polynomial_decay(learning_rate=base_learning_rate,
                                                        decay_steps=decay_steps,
                                                        end_learning_rate = end_learning_rate,
                                                        power=1.,
                                                        global_step=global_step_op,
                                                        name="polynomial_decay_power_1")
    tf.summary.scalar("polynomial_decay_power_1", learning_rate_power_1)
    learning_rate_power_05 = tf.train.polynomial_decay(learning_rate=base_learning_rate,
                                                      decay_steps=decay_steps,
                                                      end_learning_rate=end_learning_rate,
                                                      power=0.5,
                                                      global_step=global_step_op,
                                                      name="polynomial_decay_power_05")
    tf.summary.scalar("polynomial_decay_power_05", learning_rate_power_05)
    learning_rate_power_2 = tf.train.polynomial_decay(learning_rate=base_learning_rate,
                                                       decay_steps=decay_steps,
                                                       end_learning_rate=end_learning_rate,
                                                       power=2,
                                                       global_step=global_step_op,
                                                       name="polynomial_decay_power_2")
    tf.summary.scalar("polynomial_decay_power_2", learning_rate_power_2)
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