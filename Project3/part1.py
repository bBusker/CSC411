import random
from math import *
import util 

class sets:
    training = 0
    validation = 1
    test = 2

fakes, reals = util.generate_sets()

print fakes

sets_fakes = {sets.training: fakes[:909], sets.validation: fakes[909:1104], sets.test: fakes[1104:1298]}
sets_reals = {sets.training: reals[:1378], sets.validation: reals[1378:1673], sets.test: reals[1673:1968]}

priors_fakes = {"total": 0}
for headline in sets_fakes[sets.training]:
    for word in headline:
        if word in priors_fakes:
            priors_fakes[word] += 1
        else:
            priors_fakes[word] = 1
        priors_fakes["total"] += 1

priors_reals = {"total": 0}
for headline in sets_reals[sets.training]:
    for word in headline:
        if word in priors_reals:
            priors_reals[word] += 1
        else:
            priors_reals[word] = 1
        priors_reals["total"] += 1


p_fake = len(fakes)/float(len(reals)+len(fakes))
p_real = len(reals)/float(len(fakes)+len(reals))

def nb_predictor(headline, priors_reals, priors_fakes, p_fake): #TODO: train on training set, m and p paratemetrs???
    #Returns probability of the headline being fake news
    #p(y|x) = p(x|y)*p(y)/p(x)

    p_x_noty = 0
    p_x_y = 0
    p_y = p_fake

    reals_total = priors_reals["total"] #TODO: totals counting
    fakes_total = priors_fakes["total"]


    for word in headline:
        try:
            c_real = priors_reals[word]
        except:
            c_real = 1
            reals_total += 1

        try:
            c_fake = priors_fakes[word]
        except:
            c_fake = 1
            fakes_total += 1

        p_x_noty += log(c_real) - log(reals_total)
        p_x_y += log(c_fake) - log(fakes_total)

    p_yx = exp(p_x_y) * p_y / (exp(p_x_y) * p_y + exp(p_x_noty) * (1-p_y))

    return p_yx


correct = 0
print("testing on fake training set")
for i in range(len(sets_fakes[sets.test])):
    if nb_predictor(sets_fakes[sets.test][i], priors_reals, priors_fakes, p_fake) >= 0.5:
        correct += 1
print("correct: {}".format(float(correct)/len(sets_fakes[sets.test])))