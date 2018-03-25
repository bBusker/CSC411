import data_processor
import classifier
import numpy as np

fake, real = data_processor.loadHeadlines()

train, val, test, embedding, vocabstoi = classifier.prep_data(fake, real)

temp = classifier.toVariable(train, vocabstoi)

print("end")

# data_processor.generateVocabulary(fake + real)
#vocabulary = data_processor.loadVocabulary()
