from __future__ import absolute_import, division, print_function
import pwd
import sys
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np


import json

#sys.path.append('/nas1-nfs1/home/rsr200002/CodeXGLUE/Code-Code/Defect-detection/code1')

import random

#from Read_Count import get_count,read_format_code
from pathlib import Path
import math

filename = "test.jsonl"
label1 = 0
label0 = 0
label2 = 0
label3 = 0
with open(filename) as f:
        for line in f:
            js = json.loads(line.strip())
            
            label = js['label']
            if (label ==0):
                label0 = label0+1
            elif (label ==1):
                label1 = label1+1
            elif (label ==2):
                label2 = label2+1
            else:
                label3 = label3+1

print("count0", label0)
print("count1", label1)
print("count2", label2)
print("count3", label3)

            