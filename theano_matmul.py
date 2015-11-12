#!/usr/bin/env python

import time
import numpy as np
import theano as th
import theano.tensor as T

N = 10
A = th.shared(np.random.randn(N*1024,N*1024).astype(th.config.floatX), name="A")
B = th.shared(np.random.randn(N*1024,N*1024).astype(th.config.floatX), name="B")
C = T.dot(A, B)

fn = th.function([], C)
fn()

tmin = 100
for _ in range(10):
    t0 = time.time()
    C_np = np.array(fn())
    t1 = time.time()
    tmin = min(t1-t0, tmin)

print(tmin)
