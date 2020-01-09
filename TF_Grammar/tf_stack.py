#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tf_stack.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/18 下午4:56
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    t = np.random.randn(2, 8)
    s = tf.reshape(t, (-1, 2, 4))
    b = tf.unstack(s, axis=1)

    with tf.Session() as sess:
        print(t)
        print(s.eval())
        print(sess.run(b))