import random
import numpy as np
import nb_trainer
import dt_trainer

# Open and read headlines from file
f_real = open("clean_real.txt")
f_fake = open("clean_fake.txt")

reals = [str.split(line) for line in f_real]
fakes = [str.split(line) for line in f_fake]


# Randomize and generate training, validation, test sets
random.seed(0)
np.random.seed(0)
random.shuffle(reals)
random.shuffle(fakes)

sets_reals = {"train": reals[:1378], "val": reals[1378:1673], "test": reals[1673:1968]}
sets_fakes = {"train": fakes[:909], "val": fakes[909:1104], "test": fakes[1104:1298]}


# Generate Priors for naive bayes on validation set
priors_reals, priors_fakes = nb_trainer.generate_priors(sets_reals, sets_fakes, "train")
p_real = len(reals)/float(len(fakes)+len(reals))
p_fake = len(fakes)/float(len(reals)+len(fakes))


# Testing
nb_trainer.tester(sets_reals, sets_fakes, priors_reals, priors_fakes, p_fake, "test")
nb_trainer.tester(sets_reals, sets_fakes, priors_reals, priors_fakes, p_fake, "val")
nb_trainer.tester(sets_reals, sets_fakes, priors_reals, priors_fakes, p_fake, "train")

# Best Words Printouts
real_present, fake_present, real_notpresent, fake_notpresent = nb_trainer.highest_probs(priors_reals, priors_fakes, p_fake)
print("")
print("best predictors for real headline if present")
for i in range(len(real_present)):
    print("{}&{}&{:.2f}".format(i+1, real_present[i][0], real_present[i][1]))

print("")
print("best predictors for fake headline if present")
for i in range(len(fake_present)):
    print("{}&{}&{:.2f}".format(i+1, fake_present[i][0], fake_present[i][1]))

print("")
print("best predictors for real headline if absent")
for i in range(len(real_notpresent)):
    print("{}&{}&{:.2f}".format(i+1, real_notpresent[i][0], real_notpresent[i][1]))

print("")
print("best predictors for fake headline if absent")
for i in range(len(fake_notpresent)):
    print("{}&{}&{:.2f}".format(i+1, fake_notpresent[i][0], fake_notpresent[i][1]))

priors_reals, priors_fakes = nb_trainer.remove_stop_words(priors_reals, priors_fakes)
real_present, fake_present, real_notpresent, fake_notpresent = nb_trainer.highest_probs(priors_reals, priors_fakes, p_fake)
print("")
print("best predictors for real headline if present (no stop words)")
for i in range(len(real_present)):
    print("{}&{}&{:.2f}".format(i+1, real_present[i][0], real_present[i][1]))

print("")
print("best predictors for fake headline if present(no stop words)")
for i in range(len(fake_present)):
    print("{}&{}&{:.2f}".format(i+1, fake_present[i][0], fake_present[i][1]))

print("")
print("best predictors for real headline if absent(no stop words")
for i in range(len(real_notpresent)):
    print("{}&{}&{:.2f}".format(i+1, real_notpresent[i][0], real_notpresent[i][1]))

print("")
print("best predictors for fake headline if absent(no stop words)")
for i in range(len(fake_notpresent)):
    print("{}&{}&{:.2f}".format(i+1, fake_notpresent[i][0], fake_notpresent[i][1]))
