#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File collections_test.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 21/11/2019 PM 19:08

import os
import collections

if __name__ == "__main__":


    # namedtuple
    t = collections.namedtuple(typename='number', field_names=['a', 'b'])
    block = t(1, 2)
    print(block)

    s = collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])
    block = s('resnet_50', lambda x: x ** 2, [{ 'depth': 2 * 4,
                                  'depth_bottleneck': 64,
                                  'stride': 1}] * (3 - 1) + [{'depth': 64 * 4,
                                                              'depth_bottleneck': 64,
                                                              'stride': 64}])
    print(block)
    print(block.scope)
    print(block.unit_fn(2))
    print(block.args)

    # deque
    q = collections.deque([1, 3, 4])
    q.append('a')
    print(q)
    q.appendleft(0)
    print(q)

    # defaultdict
    s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
    dic = collections.defaultdict(list)
    for k, v in s:
        dic[k].append(v)
    print(dic)

    # OrderedDict
    # 相当于用列表（有序）来维护字典（无序）排序
    oder_dict = collections.OrderedDict()
    oder_dict[0] = 'essex'
    oder_dict[3] = 'folsom'
    oder_dict[4] = 'havana'
    oder_dict[2] = 'liberty'
    oder_dict[1] = 'grizzly'
    print(oder_dict)

    # Count
    cnt = collections.Counter(['red', 'blue', 'red', 'green', 'blue', 'blue'])
    print(cnt)






