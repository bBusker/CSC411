import data_processor
import numpy
from torchtext import data
import torch

templist = [["sample", "headline", "one"], ["sample", "headline", "two"]]


def prep_data():
    sentence = data.Field(
        sequential=True,
        fix_length=20,
        tokenize=data_processor.clean,
        pad_first=True,
        tensor_type=torch.LongTensor,
        lower=True
    )

    label = data.Field(
        sequential=False,
        use_vocab=False,
        tensor_type=torch.ByteTensor
    )

    fields = [('sentence_text', sentence), ('label', label)]

    headlines = []
    for temp in templist:
        headline = data.Example.fromlist((temp, 1), fields)
        headlines.append(headline)

    train = data.Dataset(headlines, fields)
    val = data.Dataset(headlines, fields)
    test = data.Dataset(headlines, fields)

    sentence.build_vocab(train, val, test,
                         max_size=100,
                         min_freq=1,
                         vectors="glove.6B.50d"
    )

    return train, val, test