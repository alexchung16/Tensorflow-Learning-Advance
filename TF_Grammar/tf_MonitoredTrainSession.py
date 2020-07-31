#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tf_MonitoredTrainSession.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/1/10 下午5:05
# @ Software   : PyCharm
#-------------------------------------------------------


import numpy as np


if __name__ == "__main__":
    np.random.seed(0)
    img_batch = np.random.random((2, 3, 3, 3))
    print(img_batch)
    print(img_batch[:, :, :, ::-1])
