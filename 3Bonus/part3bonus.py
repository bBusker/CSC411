import data_processor
import classifier

fake, real = data_processor.loadHeadlines()

temp = classifier.prep_data(fake, real)

# data_processor.generateVocabulary(fake + real)
#vocabulary = data_processor.loadVocabulary()
