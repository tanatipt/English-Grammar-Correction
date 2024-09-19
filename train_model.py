from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from nltk.translate.gleu_score import corpus_gleu
from torch.optim import AdamW
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import copy

def create_batch(X, y):
    batch = tokeniser([task_prefix + sequence for sequence in X], padding=True, truncation=True, return_tensors="pt").to(device)
    labels = tokeniser(y, padding=True, truncation=True, return_tensors="pt").to(device).input_ids
    labels[labels== tokeniser.pad_token_id] = -100
    batch['labels'] = labels
    
    return batch

class GCEDataset(Dataset):

    def __init__(self, path):
        dataset = pd.read_csv(path)
        self.X = dataset['modified'].to_list()
        self.y = dataset['sentence'].to_list()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

# Learning rate : [1e-4, 3e-4]
# Decoder Unfreeze : [1, 2]
# Encoder Unfreeze : [1, 2]
learning_rate = 1e-4
n_decoder_unfreeze = 1
n_encoder_unfreeze = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epoch_num = 100

train_data = GCEDataset("preprocessed/train.csv")
valid_data = GCEDataset("preprocessed/valid.csv")
test_data = GCEDataset("preprocessed/test.csv")

torch.manual_seed(2543673)
train_loader = DataLoader(train_data, batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size, shuffle=False)

t5_checkpoint = "google/flan-t5-base"
tokeniser = AutoTokenizer.from_pretrained(t5_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(t5_checkpoint)
model.to(device)

print(model)
for parameters in model.parameters():
    parameters.requires_grad = False

n_decoder_layers = len(model.decoder.block)

for layer in model.decoder.block[n_decoder_layers - n_decoder_unfreeze:]:
    for parameters in layer.parameters():
        parameters.requires_grad = True

n_encoder_layers = len(model.encoder.block)

for layer in model.encoder.block[n_encoder_layers- n_encoder_unfreeze:]:
    for parameters in layer.parameters():
        parameters.requires_grad=True

optimiser = AdamW(filter(lambda x : x.requires_grad, model.parameters()), lr=learning_rate)
total_steps = epoch_num * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimiser, num_warmup_steps=0, num_training_steps=total_steps)
train_stats = {"train_loss" : [], "train_gleu": [], "valid_loss" : [], "valid_gleu" : []}
task_prefix = "correct grammar: "

best_epoch = None
min_val_loss = float('inf')
best_model_param = None
patience = 5
epoch_no_update = 0

for epoch_idx in range(epoch_num):
    epoch_stats = {"train_loss" : [], "valid_loss" : []}
    train_references = []
    train_hypotheses = []
    
    with tqdm(range(len(train_loader))) as ptrain_bar:
        for X_train, y_train in train_loader:
            model.train()
            train_batch = create_batch(X_train, y_train)
            train_loss = model(**train_batch, output_attentions=False, output_hidden_states=False).loss

            optimiser.zero_grad()
            train_loss.backward()
            optimiser.step()
            scheduler.step()
            
            model.eval()
            train_output = model.generate(input_ids = train_batch['input_ids'], attention_mask = train_batch["attention_mask"])
            hypotheses = tokeniser.batch_decode(train_output, skip_special_tokens=True)
            train_references += [[sentence.split()] for sentence in y_train]
            train_hypotheses += [hypothesis.split() for hypothesis in hypotheses]

            epoch_stats["train_loss"].append(train_loss.detach().cpu().item())
            ptrain_bar.update(1)

    model.eval()
    valid_references = []
    valid_hypotheses = []
    
    with torch.no_grad(), tqdm(range(len(valid_loader))) as pvalid_bar:

        for X_valid, y_valid, in valid_loader:
            valid_batch = create_batch(X_valid, y_valid)
            valid_loss = model(**valid_batch, output_attentions=False, output_hidden_states=False).loss
            
            valid_output = model.generate(input_ids = valid_batch["input_ids"], attention_mask = valid_batch["attention_mask"])
            hypotheses = tokeniser.batch_decode(valid_output, skip_special_tokens=True)
            valid_references +=  [[sentence.split()] for sentence in y_valid]
            valid_hypotheses += [hypothesis.split() for hypothesis in hypotheses]
            
            epoch_stats['valid_loss'].append(valid_loss.cpu().item())
            pvalid_bar.update(1)

    valid_gleu = corpus_gleu(valid_references, valid_hypotheses)
    train_gleu = corpus_gleu(train_references, train_hypotheses)
    train_stats["train_gleu"].append(train_gleu)
    train_stats["valid_gleu"].append(valid_gleu)
    
    for key, value in epoch_stats.items():
        mean_val = np.mean(value)
        train_stats[key].append(mean_val)

    epoch_train_loss = train_stats["train_loss"][-1]
    epoch_valid_loss = train_stats["valid_loss"][-1]

    print("Epoch {:d} - train_loss: {:.4f}, train_gleu : {:.4f}, valid_loss: {:.4f}, valid_gleu: {:.4f}".format(epoch_idx,epoch_train_loss, train_gleu, epoch_valid_loss, valid_gleu))

    if epoch_valid_loss < min_val_loss:
      min_val_loss = epoch_valid_loss
      best_model_param = copy.deepcopy(model.state_dict())
      best_epoch = epoch_idx
      epoch_no_update = 0 
    else:
      epoch_no_update += 1 
    
    if epoch_no_update >= patience:
      break

    
model.load_state_dict(best_model_param)
model.eval()
test_losses = []
test_references = []
test_hypotheses = []

with torch.no_grad(), tqdm(range(len(test_loader))) as ptest_bar:
  for X_test, y_test in test_loader:
    test_batch = create_batch(X_test, y_test)
    test_loss = model(**test_batch, output_attentions=False, output_hidden_states=False).loss
    
    test_output = model.generate(input_ids = test_batch["input_ids"], attention_mask = test_batch["attention_mask"])
    hypotheses = tokeniser.batch_decode(test_output, skip_special_tokens=True)
    test_references +=  [[sentence.split()] for sentence in y_test]
    test_hypotheses += [hypothesis.split() for hypothesis in hypotheses]

    test_losses.append(test_loss.cpu().item())
    ptest_bar.update(1)
  
test_loss = np.mean(test_losses)
test_gleu = corpus_gleu(test_references, test_hypotheses)
test_stats = {'test_loss' : test_loss, 'test_gleu' : test_gleu}
torch.save({"best_param" : best_model_param, "best_epoch": best_epoch, "train_stats" : train_stats, "test_stats" : test_stats }, f"model/lr_{learning_rate}_n_encoder_{n_encoder_unfreeze}_n_decoder_{n_decoder_unfreeze}.pt")

