#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : cosine_decay_cycle.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/7/31 下午2:17
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import tensorflow as tf

summary_path = './summary'
method = 'cosine_decay_cycle'

max_step = 20000
base_learning_rate = 0.01
first_decay_step = 5000
alpha = 0.0001
summary_step = 10

def main():

    global_step_op = tf.train.get_or_create_global_step()

    learning_rate =  tf.train.cosine_decay_restarts(learning_rate=base_learning_rate,
                                                    first_decay_steps=first_decay_step,
                                                    t_mul = 1,
                                                    m_mul = 1,
                                                    alpha=alpha,
                                                    global_step=global_step_op,
                                                    name="cosine_decay_cycle")
    tf.summary.scalar("cosine_decay_cycle_1_1", learning_rate)
    learning_rate = tf.train.cosine_decay_restarts(learning_rate=base_learning_rate,
                                                   first_decay_steps=first_decay_step,
                                                   t_mul=2,
                                                   m_mul=1,
                                                   alpha=alpha,
                                                   global_step=global_step_op,
                                                   name="cosine_decay_cycle")
    tf.summary.scalar("cosine_decay_cycle_2_1", learning_rate)
    learning_rate = tf.train.cosine_decay_restarts(learning_rate=base_learning_rate,
                                                   first_decay_steps=first_decay_step,
                                                   t_mul=2,
                                                   m_mul=0.5,
                                                   alpha=alpha,
                                                   global_step=global_step_op,
                                                   name="cosine_decay_cycle")
    tf.summary.scalar("cosine_decay_cycle_2_05", learning_rate)
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