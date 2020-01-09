#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File run_eval.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 10/10/2019 AM 10:10

import os
import numpy
import tensorflow as tf


if __name__ == "__main__":
    # Tensor
    m = tf.constant(value=2.0, dtype=tf.float32)
    n = tf.constant(value=3.0, dtype=tf.float32)
    p = tf.multiply(n , m)
    q = tf.multiply(m , n)

    # Operation
    x = tf.placeholder(dtype=tf.float32)
    w = tf.Variable(initial_value=2.0)
    b = tf.Variable(initial_value=3.0)
    y = tf.multiply(w, x) + b

    sess = tf.Session()
    try:
        print(sess.run(m))
        # operation run
        sess.run(tf.global_variables_initializer())
        # Cannot evaluate tensor using `eval()`: No default session is registered
        # 直接调用 tensor.eval 方法，程序会抛出找不到默认会话错误
        print(m.eval())
        # Cannot execute operation using `run()`: No default session is registered
        # 直接调用 operation.run 方法，程序会抛出找不到默认会话错误
        tf.global_variables_initializer().run()

    except Exception as e:
        print(e)

    finally:
        # Tensor.eval() 等价于 tf.get.default_session().run()
        # 因此，如果要调用Tensor.eval(), 只需要将某个会话注册为默认会话，如下
        # 方法一： 使用with语句，隐式地调用Session创建会话，并注册为默认会话
        with tf.Session() as sess0:
            assert sess0 is tf.get_default_session()
            # tensor
            assert sess0.run(m) == m.eval()
            print(sess0.run(m))
            print(m.eval())
            print(sess0.run([p, q]))
            # operation
            tf.global_variables_initializer().run()
            fetch0 = sess.run(y, feed_dict={x: 1.0})
            fetch1 = y.eval(feed_dict={x: 1.0})
            print('yO={0}, y1={1}'.format(fetch0, fetch1))

            sess0.close()
        # 方法二， 将已经创建的会话，注册为默认会话
        with sess.as_default():
            assert sess is tf.get_default_session()
            # tensor
            assert sess.run(m) == m.eval()
            print(sess.run(m))
            print(m.eval())
            print(sess.run([p, q]))
            # operation
            tf.global_variables_initializer().run()
            fetch0 = sess.run(y, feed_dict={x: 1.0})
            fetch1 = y.eval(feed_dict={x: 1.0})
            print('yO={0}, y1={1}'.format(fetch0, fetch1))

            sess.close()


