

# This code implements a binary classification problem 
# It classifies the senetnces into English and Spanish 
 
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Input data
data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

#print(data)
#print(data.type)

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

# A dicionary mapping of each word in the vocab to an integer 
wordToId = {}
#print (len(wordToId))
for sent,_ in data+test_data:
    for word in sent:
        if word not in wordToId:
            wordToId[word] = len(wordToId)
print(wordToId)
VOCAB_SIZE = len(wordToId)
labels = {"SPANISH": 0, "ENGLISH": 1}
NUM_LABELS = 2 

class BoWClassifier(nn.Module):  
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)
    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)
    
def make_bow_vector(sentence, wordToId):
    vec = torch.zeros(len(wordToId))
    for word in sentence:
        vec[wordToId[word]] += 1
    return vec.view(1, -1)

def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

for param in model.parameters():
    print(param)
'''
sample = data[0]
bow_vector = make_bow_vector(sample[0], wordToId)
log_probs = model(autograd.Variable(bow_vector))
print(log_probs)
'''
print ("starts")
# Run on test data before we train, just to see a before-and-after
for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, wordToId))
    log_probs = model(bow_vec)
    print(log_probs)















































