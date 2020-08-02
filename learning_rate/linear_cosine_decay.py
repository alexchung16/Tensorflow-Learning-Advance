#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : linear_cosine_decay.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/7/31 下午1:56
# @ Software   : PyCharm
#-------------------------------------------------------

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : cosine_decay.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/7/31 上午10:41
# @ Software   : PyCharm
#-------------------------------------------------------

# SGDR: Stochastic Gradient Descent with Warm Restarts

import os
import tensorflow as tf

summary_path = './summary'
method = 'linear_cosine_decay'

max_step = 20000
base_learning_rate = 0.01
decay_steps = 10000
num_periods_1= 0.5
num_periods_4 = 4
alpha = 0.001
beta = 0.001
summary_step = 10

def main():

    global_step_op = tf.train.get_or_create_global_step()

    # global_step = min(global_step, decay_steps)
    # linear_decay = (decay_steps - global_step) / decay_steps)
    # cosine_decay = 0.5 * (
    # 1 + cos(pi * 2 * num_periods * global_step / decay_steps))
    # decayed = (alpha + linear_decay + eps_t) * cosine_decay + beta
    # decayed_learning_rate = learning_rate * decayed
    linear_cosine_decay_05 = tf.train.linear_cosine_decay(learning_rate=base_learning_rate,
                                                         decay_steps=decay_steps,
                                                         num_periods=num_periods_1,
                                                         alpha=alpha,
                                                         beta=beta,
                                                         global_step=global_step_op,
                                                         name="linear_cosine_decay_05")
    tf.summary.scalar("linear_cosine_decay_05", linear_cosine_decay_05)
    linear_cosine_decay_4 =  tf.train.linear_cosine_decay(learning_rate=base_learning_rate,
                                                   decay_steps=decay_steps,
                                                   num_periods=num_periods_4,
                                                   alpha=alpha,
                                                   beta = beta,
                                                   global_step=global_step_op,
                                                   name="linear_cosine_decay_4")
    tf.summary.scalar("linear_cosine_decay_4", linear_cosine_decay_4)
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