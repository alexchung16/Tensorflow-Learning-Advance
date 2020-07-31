#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : natural_exp_decay.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/7/31 下午3:19
# @ Software   : PyCharm
#-------------------------------------------------------

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : exponential_decay.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/7/31 上午10:29
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import tensorflow as tf

summary_path = './summary'
method = 'natural_exp_decay'

max_step = 20000
base_learning_rate = 0.01
decay_rate = 0.98
decay_steps = 2000
summary_step = 10

def main():

    global_step_op = tf.train.get_or_create_global_step()

    # decayed_learning_rate = learning_rate * exp(-decay_rate * global_step / decay_step)
    learning_rate_no_stair =  tf.train.natural_exp_decay(learning_rate=base_learning_rate,
                                                        decay_rate=decay_rate,
                                                        decay_steps=decay_steps,
                                                        staircase=False,
                                                        global_step=global_step_op,
                                                        name="natural_exp_decay_no_stair")
    tf.summary.scalar("natural_exp_decay_no_stair", learning_rate_no_stair)
    learning_rate_no_stair = tf.train.natural_exp_decay(learning_rate=base_learning_rate,
                                                        decay_rate=decay_rate,
                                                        decay_steps=decay_steps,
                                                        staircase=True,
                                                        global_step=global_step_op,
                                                        name="natural_exp_decay_use_stair")
    tf.summary.scalar("natural_exp_decay_use_stair", learning_rate_no_stair)
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