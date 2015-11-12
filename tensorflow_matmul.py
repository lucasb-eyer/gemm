#!/usr/bin/env python

import sys
import time
import numpy as np
import tensorflow as tf

device = sys.argv[1] if len(sys.argv) > 1 else "/gpu:0"
N = 10

with tf.device(device):
    A = tf.Variable(tf.random_normal((N*1024,N*1024)), name="A")
    B = tf.Variable(tf.random_normal((N*1024,N*1024)), name="B")
    C = tf.matmul(A, B)

# with tf.device('/cpu:0'):
#    D = tf.identity(C)

with tf.Session() as s:
    s.run(tf.initialize_all_variables())
    s.run(C)
    # s.run(D.op)

    tmin = 100
    for _ in range(10):
        t0 = time.time()
        s.run(C)
        # s.run(D.op)
        t1 = time.time()
        tmin = min(t1-t0, tmin)

print(tmin)
