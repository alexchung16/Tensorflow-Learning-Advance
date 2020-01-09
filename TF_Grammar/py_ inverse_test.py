#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : py_ inverse_test.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/24 下午5:15
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np

if __name__ == "__main__":

    # note first bit represent: 0->positive 1->negative
    # 3 -> 0000 0011
    # step 1 inverse: 0000 0011 -> 1111 1100
    # step 2 inverse beside first bit: 1111 1100 -> 1000 0011
    # step 3 plus 1 : 1000 0011 + 1 -> 1000 0100
    # 1000 0100 equal to -4
    print(~3)
    print(~-4)

    print(~np.array([2, 3]))
    print(~np.array([True, False]))