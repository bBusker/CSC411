import data_processor
import classifier

temp = classifier.prep_data()

fake, real = data_processor.loadHeadlines()
# data_processor.generateVocabulary(fake + real)
vocabulary = data_processor.loadVocabulary()
