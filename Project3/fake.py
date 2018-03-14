import random
import nb_trainer

# Open and read headlines from file
f_real = open("clean_real.txt")
f_fake = open("clean_fake.txt")

reals = [str.split(line) for line in f_real]
fakes = [str.split(line) for line in f_fake]


# Randomize and generate training, validation, test sets
random.seed(66)
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


def tempprinter(str):
    print("real: {}".format(priors_reals[str]/float(len(reals))))
    print("fake: {}".format(priors_fakes[str]/float(len(fakes))))