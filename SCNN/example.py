#!/usr/bin/env python
# coding=utf-8
import SSCNN
import time
import sys
import os

start = time.time()
SSCNN.train('configs/config.yaml')
SSCNN.segment('configs/config.yaml')
end = time.time()
print(end-start)
# sys.exit()
# os._exit(0)