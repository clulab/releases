import torch
import torch.nn as nn

data = [
    [
        [{'text': ['sentence_1', '#?']}, {'text': ['sentence_2', '#?']}],
        [{'text': ['sentence_1', '#*']}, {'text': ['sentence_2', '#*']}],
        [{'text': ['sentence_1', '#+']}, {'text': ['sentence_2', '#+']}],
        [{'text': ['sentence_1', '# #']}, {'text': ['sentence_2', '# #']}],
        [{'text': ['sentence_1', '[#]']}, {'text': ['sentence_2', '[#]']}],
    ], 
    [
        [{'text': ['sentence_1', '#?']}, {'text': ['sentence_2', '#?']}, {'text': ['sentence_3', '#?']}],
        [{'text': ['sentence_1', '#*']}, {'text': ['sentence_2', '#*']}, {'text': ['sentence_3', '#*']}],
        [{'text': ['sentence_1', '#+']}, {'text': ['sentence_2', '#+']}, {'text': ['sentence_3', '#+']}],
    ]
]

def test_lstm_model():
    from model import LSTMEncoder, MaxEncoder, AverageEncoder
    batch = data
    number_of_transitions  = [sum([len(y) for y in x]) for x in batch]
    number_of_sentences = [[len(y) for y in x] for x in batch]

    print(number_of_transitions)
    print(number_of_sentences)
    print([len(x) for x in number_of_sentences])
    encoded = torch.rand(19, 512)

    encoded_split = list(encoded.split([y for x in number_of_sentences for y in x]))
    lens = [y for x in number_of_sentences for y in x]


    enc1 = LSTMEncoder(512, {})
    enc2 = MaxEncoder()
    enc3 = AverageEncoder()
    e1 = enc1(encoded_split, lens)
    e2 = enc2(encoded_split, lens)
    e3 = enc3(encoded_split, lens)
    print((enc2(encoded_split, lens) - enc3(encoded_split, lens)).sum())
    dropout = nn.Dropout(0.25)
    projection = nn.Linear(512, 1)
    sigmoid = nn.Sigmoid()
    logits = projection(dropout(e2)).squeeze(dim=1)
    sizes_in_logit = [len(x) for x in number_of_sentences]
    gold = torch.zeros(logits.shape)
    gold[[0] + sizes_in_logit[:-1]] = 1
    print(gold)
    print(list(logits.split(sizes_in_logit)))

test_lstm_model()