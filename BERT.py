# BERT imports
import tensorflow as tf
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

# Converting dataset to tokens that work with BERT
dfTrain = pd.read_csv("authors.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
dfTest = pd.read_csv("testInputFile.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
authorNums = {}
num = 0
for x in dfTest.label.values:
  if x not in authorNums.keys():
    authorNums[x] = num
    num+=1
# Add tokens on start and end of sentence for BERT
def tokenize(df):
  # Create sentence and label lists
  sentences = df.sentence.values
  labels = [authorNums.get(x) for x in df.label.values]
  sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
  return [tokenizer.tokenize(sent) for sent in sentences], labels

trainTokenized, trainLabels = tokenize(dfTrain)
testTokenized, testLabels = tokenize(dfTest)

# Max sequence length set to 128 as tweets can only be 280 characters and more than 128 words is highly unlikely
MAX_LEN = 128
# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
trainInputs = [tokenizer.convert_tokens_to_ids(x) for x in trainTokenized]
testInputs = [tokenizer.convert_tokens_to_ids(x) for x in testTokenized]
# Pad our input tokens
trainInputs = pad_sequences(trainInputs, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
testInputs = pad_sequences(testInputs, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
trainAttentionMasks = []
testAttentionMasks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in trainInputs:
  seq_mask = [float(i>0) for i in seq]
  trainAttentionMasks.append(seq_mask)
for seq in testInputs:
  seq_mask = [float(i>0) for i in seq]
  testAttentionMasks.append(seq_mask)

trainInputs = torch.tensor(trainInputs)
testInputs = torch.tensor(testInputs)
trainLabels = torch.tensor(trainLabels)
testLabels = torch.tensor(testLabels)
trainAttentionMasks = torch.tensor(trainAttentionMasks)
testAttentionMasks = torch.tensor(testAttentionMasks)


# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
batchSize = 32

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory
trainData = TensorDataset(trainInputs, trainAttentionMasks, trainLabels)
trainSampler = RandomSampler(trainData)
trainDataloader = DataLoader(trainData, sampler=trainSampler, batch_size=batchSize)

testData = TensorDataset(testInputs, testAttentionMasks, testLabels)
testSampler = SequentialSampler(testData)
testDataloader = DataLoader(testData, sampler=testSampler, batch_size=batchSize)


# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# This variable contains all of the hyperparemeter information our training loop needs
optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    predFlat = np.argmax(preds, axis=1).flatten()
    labelsFlat = labels.flatten()
    return np.sum(predFlat == labelsFlat) / len(labelsFlat)

t = [] 

# Store our loss and accuracy for plotting
trainLossSet = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):
  
  
  # Training
  
  # Set our model to training mode (as opposed to evaluation mode)
  model.train()
  
  # Tracking variables
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  
  # Train the data for one epoch
  for step, batch in enumerate(trainDataloader):
    print("train epoch ", step)
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Clear out the gradients (by default they accumulate)
    optimizer.zero_grad()
    # Forward pass
    loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
    trainLossSet.append(loss.item())    
    # Backward pass
    loss.backward()
    # Update parameters and take a step using the computed gradient
    optimizer.step()
    
    
    # Update tracking variables
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1

  print("Train loss: {}".format(tr_loss/nb_tr_steps))
    
    
  # Validation

  # Put model in evaluation mode to evaluate loss on the validation set
  model.eval()

  # Tracking variables 
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  # Evaluate data for one epoch
  for batch in validation_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

  print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))