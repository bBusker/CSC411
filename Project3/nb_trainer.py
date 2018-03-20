from math import *
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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

        p_x_given_real += log(c_real + m*(1-p_hat)) - log(reals_total + m)
        p_x_given_fake += log(c_fake + m*p_hat) - log(fakes_total + m)


    for word in priors_reals:
        if word in headline:
            continue
        else:
            p_x_given_real += log((reals_total + m) - (priors_reals[word] + m*(1-p_hat))) - log(reals_total + m)

    for word in priors_fakes:
        if word in headline:
            continue
        else:
            p_x_given_fake += log((fakes_total + m) - (priors_fakes[word] + m * p_hat)) - log(fakes_total + m)

    p_yx = exp(p_x_given_fake) * p_fake / (exp(p_x_given_fake) * p_fake + exp(p_x_given_real) * (1-p_fake))

    return p_yx


def word_prob(word, priors_reals, priors_fakes, p_fake, real=False, presence=True):
    # Returns probability of word predicting whether or not a headline is fake/real

    m = 2 # Virtual Examples
    p_hat = 0.6 # Virtual Example Prior (of being fake)

    p_x_given_real = 0
    p_x_given_fake = 0

    reals_total = priors_reals["count_total"]
    fakes_total = priors_fakes["count_total"]

    try:
        c_real = priors_reals[word]
    except:
        c_real = 0

    try:
        c_fake = priors_fakes[word]
    except:
        c_fake = 0

    if presence:
        p_x_given_real += log(c_real + m*(1-p_hat)) - log(reals_total + m)
        p_x_given_fake += log(c_fake + m*p_hat) - log(fakes_total + m)
    elif not presence:
        p_x_given_real += log((reals_total+m) - (c_real + m*(1-p_hat))) - log(reals_total + m)
        p_x_given_fake += log((fakes_total+m) - (c_fake + m*p_hat)) - log(fakes_total + m)

    if not real:
        p_yx = exp(p_x_given_fake) * p_fake
    elif real:
        p_yx = exp(p_x_given_real) * (1-p_fake)

    p_yx /= (exp(p_x_given_fake) * p_fake + exp(p_x_given_real) * (1-p_fake))

    return p_yx


def highest_probs(priors_reals, priors_fakes, p_fake):
    word_probs = {}
    for word in priors_reals:
        word_probs[word] = word_prob(word, priors_reals, priors_fakes, p_fake, real=True)
    for word in priors_fakes:
        word_probs[word] = word_prob(word, priors_reals, priors_fakes, p_fake, real=True)
    word_probs.pop("count_total")
    word_probs_real_present = sorted(word_probs.items(), key=lambda (x): x[1], reverse=True)

    word_probs = {}
    for word in priors_reals:
        word_probs[word] = word_prob(word, priors_reals, priors_fakes, p_fake)
    for word in priors_fakes:
        word_probs[word] = word_prob(word, priors_reals, priors_fakes, p_fake)
    word_probs.pop("count_total")
    word_probs_fake_present = sorted(word_probs.items(), key=lambda (x): x[1], reverse=True)

    word_probs = {}
    for word in priors_reals:
        word_probs[word] = word_prob(word, priors_reals, priors_fakes, p_fake, real=True, presence=False)
    for word in priors_fakes:
        word_probs[word] = word_prob(word, priors_reals, priors_fakes, p_fake, real=True, presence=False)
    word_probs.pop("count_total")
    word_probs_real_notpresent = sorted(word_probs.items(), key=lambda (x): x[1], reverse=True)

    word_probs = {}
    for word in priors_reals:
        word_probs[word] = word_prob(word, priors_reals, priors_fakes, p_fake, presence=False)
    for word in priors_fakes:
        word_probs[word] = word_prob(word, priors_reals, priors_fakes, p_fake, presence=False)
    word_probs.pop("count_total")
    word_probs_fake_notpresent = sorted(word_probs.items(), key=lambda (x): x[1], reverse=True)

    return word_probs_real_present[0:10], word_probs_fake_present[0:10], word_probs_real_notpresent[0:10], word_probs_fake_notpresent[0:10]


def remove_stop_words(priors_reals, priors_fakes):
    for word in ENGLISH_STOP_WORDS:
        if word in priors_reals:
            del(priors_reals[word])
        if word in priors_fakes:
            del(priors_fakes[word])

    return priors_reals, priors_fakes


def tester(sets_reals, sets_fakes, priors_reals, priors_fakes, p_fake, testing_set):
    # Tests the naive bayes algorithm on the specified sets using the input priors

    correct = 0

    print("Testing Naive Bayes Classifier on {} Set ".format(testing_set.title()))

    for i in range(len(sets_reals[testing_set])):
        prediction = predictor(sets_reals[testing_set][i], priors_reals, priors_fakes, p_fake)
        if prediction < 0.5:
            correct += 1
        if prediction > 1 or prediction < 0:
            print("INVALID PREDICTION: {}".format(prediction))

    for i in range(len(sets_fakes[testing_set])):
        prediction = predictor(sets_fakes[testing_set][i], priors_reals, priors_fakes, p_fake)
        if prediction >= 0.5:
            correct += 1
        if prediction > 1 or prediction < 0:
            print("INVALID PREDICTION: {}".format(prediction))

    print("--Correct: {:.2f}%".format(float(correct)/len(sets_fakes[testing_set] + sets_reals[testing_set])*100))