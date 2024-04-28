# -*- coding: utf-8 -*-

# deepfake detection -- late fusion model

#!find . -type d -name "acv-presidential-dataset"

#!pip install torchaudio

#!pip install librosa

# imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os
import torch
import torchaudio
import librosa
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoProcessor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from transformers import BertModel, BertTokenizer, AutoTokenizer, Wav2Vec2Model, DistilBertModel, DistilBertConfig, BertLayer
from transformers import Trainer, TrainingArguments, AdamW
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

data_dir_real = "./acv-presidential-dataset/real"
data_dir_fake = "./acv-presidential-dataset/fake"

# create dataframe containing audio, text, and screenshot images for each data sample
data = []

# real data
presidents = ['biden', 'trump']
for name in presidents:
  for i in range(1,16):
    audio_file = name + '-real-' + str(i) + '-audio.wav'
    audio_path = os.path.join(data_dir_real, 'audio-files')
    text_file = name + '-real-' + str(i) + '.txt'
    text_path = os.path.join(data_dir_real, 'text-files')
    screenshot_folder_name = name + '-real-' + str(i)
    screenshot_folder_path = os.path.join(data_dir_real, 'screenshots', screenshot_folder_name)
    image_files = os.listdir(screenshot_folder_path)
    sample = {'audio': audio_file, 'text': text_file, 'images': image_files, 'audio_path': audio_path, 'text_path': text_path, 'image_path': screenshot_folder_path, 'label': 0}
    data.append(sample)

# deepfake data
for name in presidents:
  for i in range(1,9):
    audio_file = name + '-fake-' + str(i) + '.wav'
    audio_path = os.path.join(data_dir_fake, 'audio-files')
    text_file = name + '-fake-' + str(i) + '.txt'
    text_path = os.path.join(data_dir_fake, 'text-files')
    screenshot_folder_name = name + '-fake-' + str(i)
    screenshot_folder_path = os.path.join(data_dir_fake, 'screenshots', screenshot_folder_name)
    image_files = os.listdir(screenshot_folder_path)
    sample = {'audio': audio_file, 'text': text_file, 'images': image_files, 'audio_path': audio_path, 'text_path': text_path, 'image_path': screenshot_folder_path, 'label': 1}
    data.append(sample)

data_df = pd.DataFrame(data)

print(data_df)

# sanity check
print(data_df.loc[1,'image_path'])

# custom dataset

class CustomDataset(Dataset):
  def __init__(self, data_df):
    self.data_df = data_df
    self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

  def __len__(self):
    return len(self.data_df)

  def __getitem__(self, idx):
    audio = self.data_df.loc[idx, 'audio']
    text = self.data_df.loc[idx, 'text']
    images = self.data_df.loc[idx, 'images']
    label = self.data_df.loc[idx, 'label']

    audio_path = self.data_df.loc[idx, 'audio_path']
    full_audio_path = os.path.join(audio_path, audio)
    opened_audio, sample_rate = librosa.load(full_audio_path, sr=None)

    text_path = self.data_df.loc[idx, 'text_path']
    full_text_path = os.path.join(text_path, text)
    with open(full_text_path, 'r') as file:
      opened_text = file.read()

    image_path = self.data_df.loc[idx, 'image_path']
    processed_images = []

    for img in images:
      full_image_path = os.path.join(image_path, img)
      image = Image.open(full_image_path).convert("RGB")
      processed_image = self.image_processor(images=image, return_tensors="pt")
      processed_image = processed_image["pixel_values"]
      processed_images.append(processed_image)

    item = {
        "audio": opened_audio,
        "sample_rate": sample_rate,
        "text": opened_text,
        "images": processed_images,
        "label": label
    }

    return item

# custom collate function

text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
audio_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

def collate_fn(batch):
  audios = [item['audio'] for item in batch]
  sample_rates = [item['sample_rate'] for item in batch]
  texts = [item['text'] for item in batch]
  images_per_item = [item['images'] for item in batch] # list of lists of tensors 
  labels = [item['label'] for item in batch]

  resampled_audios = []
  for audio, sample_rate in zip(audios, sample_rates):
      audio_tensor = torch.tensor(audio)
      audio_resampled = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio_tensor)
      resampled_audios.append(audio_resampled)

  resampled_audios = pad_sequence(resampled_audios, batch_first=True)

  processed_audios = []
  for audio_resampled in resampled_audios:
      inputs = audio_processor(audio_resampled, sampling_rate=None, return_tensors="pt")
      processed_audios.append(inputs.input_values.unsqueeze(0))

  processed_audios = torch.stack(processed_audios, dim=0)

  # process each item's images as a separate batch (return list instead of tensor)
  batched_images = [torch.cat(images, dim=0).squeeze(1) for images in images_per_item]

  tokenized_texts = text_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
  tokenized_texts = tokenized_texts['input_ids']
  text_attention_mask = (tokenized_texts != text_tokenizer.pad_token_id).long()

  labels = torch.LongTensor(labels)

  batch_dict = {
      "audios": processed_audios,
      "texts": tokenized_texts,
      "images": batched_images,
      'text_attention_mask': text_attention_mask,
      "labels": labels
  }

  return batch_dict

# create training, validation, and testing dataframes

train_df, test_df = train_test_split(data_df, test_size=0.25, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# reset indices of dataframes to work in dataset class

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# create training, validation, and testing datasets using custom dataset class

train_dataset = CustomDataset(train_df)
val_dataset = CustomDataset(val_df)
test_dataset = CustomDataset(test_df)

# create training, validation, and testing data loaders
# added 'pin_memory' argument to help prevent memory errors

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True, collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, pin_memory=True, collate_fn=collate_fn, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, pin_memory=True, collate_fn=collate_fn, num_workers=0)

print(len(train_loader)) 
print(len(val_loader)) 
print(len(test_loader))

# smaller wav2vec2 model to avoid memory issues

class SimplifiedWav2Vec2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=10, stride=5, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=10, stride=5, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=10, stride=5, padding=3),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=10, stride=5, padding=3),
            nn.ReLU(),
            nn.Conv1d(512, 768, kernel_size=10, stride=5, padding=3),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        return x

# train the smaller wav2vec2 model

def train_wav2vec2():
    wav2vec2_small = SimplifiedWav2Vec2Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(wav2vec2_small.parameters(), lr=0.001)

    # training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        wav2vec2_small.train()
        running_loss = 0.0
        for batch in train_loader:
            audios = batch['audios']
            audios = audios.squeeze(1)
            labels = batch['labels']
            optimizer.zero_grad()
            outputs = wav2vec2_small(audios)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * audios.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return wav2vec2_small

wav2vec2_small = train_wav2vec2() # trained smaller model

# save weights of trained simplified wav2vec2 model (only need to do this once)
torch.save(wav2vec2_small.state_dict(), './smaller_model_weights.pth')

# print the keys of the loaded checkpoint dictionary --> sanity check
checkpoint = torch.load('smaller_model_weights.pth')
print(checkpoint.keys())

# load weights of trained simplified wav2vec2 model (so that i only have to train it once)
wav2vec2_small = SimplifiedWav2Vec2Model()
wav2vec2_small.load_state_dict(torch.load('./smaller_model_weights.pth'))

# print the state dictionary --> sanity check
print(wav2vec2_small.state_dict())

# smaller distilbert model to avoid memory issues --> only use if loading weights fails

class SmallerDistilBERT(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        # create a new transformer stack with only the first 4 layers
        new_layers = nn.ModuleList([layer for i, layer in enumerate(self.transformer.layer) if i < 4])

        # replace the original transformer layers with the new ones
        self.transformer.layer = new_layers
        self.transformer.num_layers = len(new_layers)

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # use the modified transformer with only the first 4 layers
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

# create a smaller bert model and load it with weights from pre-trained model

class SmallerDistilBERT(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        # create a new transformer stack with only the first 4 layers
        new_layers = nn.ModuleList([layer for i, layer in enumerate(self.transformer.layer) if i < 4])

        # replace the original transformer layers with the new ones
        self.transformer.layer = new_layers
        self.transformer.num_layers = len(new_layers)

# load the original DistilBERT model
original_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

# create an instance of SmallerDistilBERT
config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
distilbert_small = SmallerDistilBERT(config)

# copy the parameters from the original model to the smaller model (only for the appropriate layers)
for name, param in original_bert.named_parameters():
    if name.startswith('distilbert.transformer') and int(name.split('.')[3]) < 4:
        distilbert_small.state_dict()[name].copy_(param.data)

# save the smaller model
torch.save(distilbert_small.state_dict(), './small_distilbert_weights.pth')

# load the smaller model
distilbert_small.load_state_dict(torch.load('./small_distilbert_weights.pth'))

torch.cuda.empty_cache()

# late fusion model using audio, images, and text
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LateFusionModel(nn.Module):
  def __init__(self, audio_encoder, text_encoder, resnet_layers=5, hidden_size=768, num_classes=2, freeze=False):
      super().__init__()

      self.audio_encoder = audio_encoder
      self.audio_linear = nn.Linear(768, hidden_size)

      self.text_encoder = text_encoder

      resnet = models.resnet50(pretrained=True)
      self.image_encoder = nn.Sequential(*list(resnet.children())[:resnet_layers])
      self.image_linear = nn.Linear(256 * 56 * 56, hidden_size)

      self.classifier = nn.Sequential(
          nn.Linear(hidden_size * 3, 512),
          nn.ReLU(),
          nn.Dropout(0.1),
          nn.Linear(512, num_classes)
      )

      if freeze:
        for param in self.image_encoder.parameters():
          param.requires_grad = False
        for param in self.text_encoder.parameters():
          param.requires_grad = False
        for param in self.audio_encoder.parameters():
          param.requires_grad = False

      self.normalization = nn.LayerNorm(hidden_size * 3)

  def forward(self, audios, images, texts, text_attention_mask):
      # create image embeddings
      embeddings = []
      # iterate through lists of images
      for imgs in images:
        imgs = imgs.to(device) # move to device here 
        # normalize images
        imgs = self.normalize_images(imgs)
        image_embeddings = self.image_encoder(imgs)
        image_embeddings = image_embeddings.view(image_embeddings.size(0), -1)
        image_embedding = image_embeddings.mean(dim=0, keepdim=True) # take mean
        embeddings.append(image_embedding)

      embeddings_tensor = torch.cat(embeddings, dim=0).to(device)
      final_image_embeddings = self.image_linear(embeddings_tensor)

      # create text embeddings
      text_embeddings = self.text_encoder(texts, attention_mask=text_attention_mask)[0][:,0,:]

      # create audio embeddings
      audio_embeddings = self.audio_encoder(audios)
      audio_embeddings = self.audio_linear(audio_embeddings)

      # concatenate image, text, and audio embeddings
      combined_embeddings = torch.cat((final_image_embeddings, text_embeddings, audio_embeddings), dim=1)

      # normalize combined embeddings
      normalized_embeddings = self.normalization(combined_embeddings)

      # classifier layer
      output = self.classifier(normalized_embeddings)

      return output

  def normalize_images(self, images):
      # normalize images to range [0, 1]
      images = images.float() / 255.0
      return images

# training and validation loop

# initialize model, criterion, and optimizer
model = LateFusionModel(audio_encoder=wav2vec2_small, text_encoder=distilbert_small)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-6)

# initialize variables for tracking best model and early stopping
num_epochs = 10
early_stop_count = 0
best_acc = 0.0
best_model_state = None

# reduce precision training to save memory
scaler = GradScaler()

for epoch in range(num_epochs):
    # training loop
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for batch in train_loader:
        audios = batch['audios']
        audios = audios.squeeze(1).to(device)
        texts = batch['texts'].to(device)
        images = batch['images'] # can't use .to(device) on list --> this is done in the forward function
        text_attention_mask = batch['text_attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(audios, images, texts, text_attention_mask)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, preds = torch.max(outputs.data, 1)
        running_corrects += (preds == labels).sum().item()
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}')

    # validation loop
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            audios = batch['audios']
            audios = audios.squeeze(1).to(device)
            texts = batch['texts'].to(device)
            images = batch['images'] # can't use .to(device) on list --> this is done in the forward function
            text_attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(audios, images, texts, text_attention_mask)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs.data, 1)
            running_corrects += (preds == labels).sum().item()
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.4f}')

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_state = model.state_dict()  # save the state dictionary of the best model
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count >= 8:
            print(f'Early stopping at epoch {epoch + 1} to prevent overfitting')
            break

    # save the best model's state dictionary to a file after each epoch
    torch.save(best_model_state, "./best_late_fusion_model.pth")
    # empty cache after each epoch to save memory
    torch.cuda.empty_cache()

print("")
print('*'*50)
print("")
print(f'Best val Acc: {best_acc:4f}')

# save the best model's state dictionary to a file
torch.save(best_model_state, "./best_late_fusion_model.pth")

# clear cache to save memory
torch.cuda.empty_cache()

# load the best model state dictionary for testing
model.load_state_dict(torch.load("./best_late_fusion_model.pth"))
model.eval()

# testing loop
test_running_loss = 0.0
test_running_corrects = 0
test_total = 0
test_predictions = []
test_targets = []

with torch.no_grad():
    for batch in test_loader:
        audios = batch['audios']
        audios = audios.squeeze(1).to(device)
        texts = batch['texts'].to(device)
        images = batch['images'] # can't use .to(device) on list --> this is done in the forward function
        text_attention_mask = batch['text_attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(audios, images, texts, text_attention_mask)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)
        test_predictions.extend(preds.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())

        test_running_corrects += (preds == labels).sum().item()
        test_running_loss += loss.item() * images.size(0)
        test_total += labels.size(0)

# calculate test loss, accuracy, precision, and recall
test_loss = test_running_loss / test_total
test_accuracy = test_running_corrects / test_total
test_precision = precision_score(test_targets, test_predictions, average='weighted')
test_recall = recall_score(test_targets, test_predictions, average='weighted')

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
