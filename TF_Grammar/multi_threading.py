#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : multi_threading.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/5/23 下午2:51
# @ Software   : PyCharm
#-------------------------------------------------------


import threading
import tensorflow as tf

def thread_job():
    print(threading.active_count())
    print(threading.enumerate())
    print("This is a new Thread, number is %s" % threading.current_thread())


def main():
    print(threading.active_count())
    print(threading.enumerate())
    add_thread = threading.Thread(target=thread_job)
    add_thread.start()

if __name__ == "__main__":

    tf.nn.l2_normalize()

