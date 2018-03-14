import random
from math import *

def generate_sets():
    random.seed(1)
    
    f_fake = open("clean_fake.txt")
    f_real = open("clean_real.txt")

    fakes = [str.split(line) for line in f_fake]
    reals = [str.split(line) for line in f_real]

    random.shuffle(fakes)
    random.shuffle(reals)
    
    return fakes, reals