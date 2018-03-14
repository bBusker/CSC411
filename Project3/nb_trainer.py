from math import *

def generate_priors(sets_reals, sets_fakes, set_choice):
    # Generates trained sets for our prior prediction

    priors_reals = {"count_total": 0}
    for headline in sets_reals[set_choice]:
        for word in headline:
            if word in priors_reals:
                priors_reals[word] += 1
            else:
                priors_reals[word] = 1
            priors_reals["count_total"] += 1

    priors_fakes = {"count_total": 0}
    for headline in sets_fakes[set_choice]:
        for word in headline:
            if word in priors_fakes:
                priors_fakes[word] += 1
            else:
                priors_fakes[word] = 1
            priors_fakes["count_total"] += 1

    return priors_reals, priors_fakes


def predictor(headline, priors_reals, priors_fakes, p_fake):
    # Returns probability of the headline being fake news
    # p(y|x) = p(x|y)*p(y)/(p(x|y)*p(y) + p(x|~y)*p(~y))

    m = 2 # Virtual Examples
    p_hat = 0.6 # Virtual Example Prior (of being fake)

    p_x_given_real = 0
    p_x_given_fake = 0

    reals_total = priors_reals["count_total"]
    fakes_total = priors_fakes["count_total"]

    for word in headline:
        try:
            c_real = priors_reals[word]
        except:
            c_real = 0

        try:
            c_fake = priors_fakes[word]
        except:
            c_fake = 0

        p_x_given_real += log(c_real + m*p_hat) - log(reals_total + m)
        p_x_given_fake += log(c_fake + m*p_hat) - log(fakes_total + m)

    p_yx = exp(p_x_given_fake) * p_fake / (exp(p_x_given_fake) * p_fake + exp(p_x_given_real) * (1-p_fake))

    return p_yx


def tester(sets_reals, sets_fakes, priors_reals, priors_fakes, p_fake, testing_set):
    # Tests the naive bayes algorithm on the specified sets using the input priors

    correct = 0

    print("Testing Naive Bayes Classifier on {} Set ".format(testing_set.title()))

    for i in range(len(sets_reals[testing_set])):
        prediction = predictor(sets_reals[testing_set][i], priors_reals, priors_fakes, p_fake)
        if prediction < 0.5:
            correct += 1
        if prediction >= 1 or prediction < 0:
            print("INVALID PREDICTION: {}".format(prediction))

    for i in range(len(sets_fakes[testing_set])):
        prediction = predictor(sets_fakes[testing_set][i], priors_reals, priors_fakes, p_fake)
        if prediction >= 0.5:
            correct += 1
        if prediction >= 1 or prediction < 0:
            print("INVALID PREDICTION: {}".format(prediction))

    print("--Correct: {:.7f}%".format(float(correct)/len(sets_fakes[testing_set] + sets_reals[testing_set])*100))