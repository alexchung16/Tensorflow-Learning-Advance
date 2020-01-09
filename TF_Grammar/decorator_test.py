#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File decorator.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 29/10/2019 AM 10:19

import tensorflow.contrib.slim as slim


if __name__ == "__main__":


    # nor parameter
    def funA(fn):
        print('function A')
        fn()
        return 'OK'

    @funA
    def funB():
        print('function B')

    print(funB)


    # parameter
    def foo_square(fn):
        print('function foo')
        def decorator(n):
            f = fn(n) ** 2
            print('function decorator', f)
            return f
        return decorator
    @foo_square
    def test(n):
        print('function test', n)
        return n
    test(10)

    # multiple parameter
    def foo_plus(fn):

        print('function foo_plus')
        def decorator(*args, **kwargs):
            f = fn(*args, **kwargs) ** 2
            print('function decorator', f)
            return f
        return decorator

    @foo_plus
    def test_plus(a, b):
        print('function test plus')
        return a+b

    test_plus(3, 4)

    @foo_square
    @foo_plus
    def test_square_plus(a):
        print('function test plus')
        return a

    test_square_plus(3)

