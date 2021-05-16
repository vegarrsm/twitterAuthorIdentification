    

# BERT imports
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn


#Switched to cpu when comparing performance
gpu = True if torch.cuda.is_available() else False
device = torch.device("cuda" if gpu else "cpu")

if gpu:
      n_gpu = torch.cuda.device_count()
  torch.cuda.get_device_name(0)

# verify GPU availability
if gpu:
      device_name = tf.test.gpu_device_name()
  if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
  print('Found GPU at: {}'.format(device_name))

  import time

# Converting dataset to tokens that work with BERT
dfTrain = pd.read_csv("authors.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
dfTest = pd.read_csv("testInputFile.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

setupTime = time.time()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
authorNums = {}
num = 0

for x in dfTest.label.values:
      if x not in authorNums.keys():
        authorNums[x] = num
    num+=1


# Max sequence length set to 128 as tweets can only be 280 characters and more than 128 words is highly unlikely
MAX_LEN = 128

# Add tokens on start and end of sentence for BERT
def tokenize(df):
      # Create sentence and label lists
  sentences = df.sentence.values
  labels = [authorNums.get(x) for x in df.label.values]
  sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
  sentences = [tokenizer.tokenize(sent) for sent in sentences]
  sentences = [tokenizer.convert_tokens_to_ids(sent) for sent in sentences]
  sentences = pad_sequences(sentences, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
  return sentences, labels

inputs = tokenize(dfTrain)

trainInputs,testInputs, trainLabels, testLabels = train_test_split(inputs[0], inputs[1], 
                                                            random_state=2018, test_size=0.1)

# Create attention masks
attentionMasks = []
for seq in inputs[0]:
      seq_mask = [float(i>0) for i in seq]
  attentionMasks.append(seq_mask) 
trainAttentionMasks = []
testAttentionMasks = []
trainAttentionMasks, testAttentionMasks, _, _ = train_test_split(attentionMasks, inputs[0],
                                             random_state=2021, test_size=0.1)


trainInputs = torch.tensor(trainInputs)
testInputs = torch.tensor(testInputs)
testLabels = torch.tensor(testLabels)
trainLabels = torch.tensor(trainLabels)
trainAttentionMasks = torch.tensor(trainAttentionMasks)
testAttentionMasks = torch.tensor(testAttentionMasks)



# 16 and 32 was tried, but 16 delivered best results
batchSize = 16

# Using DataLoader iterator as it uses memory better than list
trainData = TensorDataset(trainInputs, trainAttentionMasks, trainLabels)
trainSampler = RandomSampler(trainData)
trainDataloader = DataLoader(trainData, sampler=trainSampler, batch_size=batchSize)


testData = TensorDataset(testInputs, testAttentionMasks, testLabels)
testSampler = SequentialSampler(testData)
testDataloader = DataLoader(testData, sampler=testSampler, batch_size=batchSize)

# Load pretrained BERT
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(authorNums))
model.cuda() if gpu else model.to(device)

paramOptimizer = list(model.named_parameters())
noDecay = ['bias', 'gamma', 'beta']
optimizerGroupedParameters = [
    {'params': [p for n, p in paramOptimizer if not any(nd in n for nd in noDecay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in paramOptimizer if any(nd in n for nd in noDecay)],
     'weight_decay_rate': 0.0}
]

# hyperparamaters
optimizer = BertAdam(optimizerGroupedParameters, lr=2e-5, warmup=.1)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    predFlat = np.argmax(preds, axis=1).flatten()
    labelsFlat = labels.flatten()
    return np.sum(predFlat == labelsFlat) / len(labelsFlat)

t = [] 

# Store our loss and accuracy for plotting
trainLossSet = []
setupTime = time.time() - setupTime


# results improved from 2 to 4 epochs, but stagnated with more
epochs = 4
trainTime = time.time()
epochTimes = []
# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):
      
  epochTime = time.time()
  
  # Training
  model.train()
  
  # Tracking variables
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  
  # Train the data for one epoch
  for step, batch in enumerate(trainDataloader):
        # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from dataloader
    dataInputIds, dataInputMasks, dataLabels = batch
    optimizer.zero_grad()
    # Forward pass
    loss = model(dataInputIds,  attention_mask=dataInputMasks, labels=dataLabels)
    trainLossSet.append(loss.item())
    # Backward pass
    loss.backward()

    optimizer.step()
    
    
    # Update tracking variables
    tr_loss += loss.item()
    nb_tr_examples += dataInputIds.size(0)
    nb_tr_steps += 1

  print("Train loss: {}".format(tr_loss/nb_tr_steps))
    
  trainTime = time.time() - trainTime

  validationTime = time.time()
    
  # Validation during training

  # Put model in evaluation mode to evaluate loss on the validation set
  model.eval()

  # Tracking variables 
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  # Evaluate data for one epoch
  for batch in testDataloader:
        # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    dataInputIds, dataInputMasks, dataLabels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits = model(dataInputIds, token_type_ids=None, attention_mask=dataInputMasks)
    
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = dataLabels.to('cpu').numpy()

    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

  validationTime = time.time() - validationTime
  epochTimes.append(time.time()-epochTime)
  print("\nvalidation time: ",validationTime,"\nvalidation time: ", trainTime)
  print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

trainTime = time.time() - trainTime
print("setupTime: --- %s seconds ---" % setupTime, "\ntrainTime: --- %s seconds ---" %trainTime, "\nepochTimes: --- ",epochTimes," seconds ---\n")


df = pd.read_csv("testInputFile.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

validationTime = time.time()
# Create sentence and label lists
sentences = df.sentence.values

# BERT tokens at start and end of each sentence
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

MAX_LEN = 128

# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
      seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask) 


prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor([authorNums.get(x) for x in labels])

#redundant?  
batchSize = 16


prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batchSize)

# Prediction on test set

# Evaluation during training
model.eval()

# Tracking variables 
predictions , true_labels = [], []
# Predict 
for batch in prediction_dataloader:
      # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  # Unpack the inputs from our dataloader
  dataInputIds, dataInputMask, dataLabels = batch
  with torch.no_grad():
    # Forward pass, calculate logit predictions
    logits = model(inputIdsTraining, token_type_ids=None, attention_mask=dataInputMask)

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = dataLabels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)
validationTime = time.time() - validationTime


cm = []
for i in range(len(true_labels)):
      cm.append(np.argmax(predictions[i], axis=1).flatten())
print(true_labels, len(authorNums.keys()))
print(cm)



#Merging sublists to one list
trueL = []
pred = []
for subList in true_labels:
  for x in subList:
    trueL.append(x)
for subList in cm:
  for x in subList:
    pred.append(x)

confusion = confusion_matrix(trueL, pred, [i for i in range(len(authorNums.keys()))])

#Calculate percentage of correct guesses¨
correct = []
for i in range(len(trueL)):
  correct.append(1) if trueL[i] == pred[i] else correct.append(0)

print("average correct guesses: ",sum(correct)/len(correct))
print(validationTime)
seaborn.heatmap(confusion, annot=True, yticklabels = authorNums.keys(), xticklabels = authorNums.keys())