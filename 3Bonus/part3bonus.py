import data_processor
import classifier
import numpy as np

# temp = classifier.prep_data()

fake, real = data_processor.loadHeadlines()
# data_processor.generateVocabulary(fake + real)
vocabulary = data_processor.loadVocabulary()

print(np.mean([len(x) for x in fake + real]))

# classifier.prep_data(fake, real)
