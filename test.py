import math
import torch
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from train import LSTMModel
import jieba
import random
from train import vector2word
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
word_vector_dict={}
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.fc(lstm_out[:, -1, :])
        return y_pred
    
def load_model(model_path, input_dim=200, hidden_dim=200, output_dim=200):
    model = LSTMModel(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def vector_sqrtlen(vector):
    return math.sqrt(torch.sum(vector ** 2))

def vector_cosine(v1, v2):
    return torch.dot(v1, v2) / (vector_sqrtlen(v1) * vector_sqrtlen(v2))

def vector2word(vector):
    max_cos = -10000
    match_word = ''
    for word, v in word_vector_dict.items():
        v = v.to(device)
        cosine = vector_cosine(vector, v)
        if cosine > max_cos:
            max_cos = cosine
            match_word = word
    return match_word, max_cos

def prepare_initial_sequence(initial_words, word_vector_dict):
    seq = [word_vector_dict[word] for word in initial_words if word in word_vector_dict]
    if len(seq) < 20:
        raise ValueError("Initial sequence is shorter than 20 words.")
    return torch.stack(seq[:20])  # Use only the last 20 words if more are provided

def generate_words(model, initial_sequence, word_vector_dict, num_words=500):
    generated_words = []
    current_seq = initial_sequence.unsqueeze(0) if initial_sequence.dim() == 2 else initial_sequence
    for _ in range(num_words):
        with torch.no_grad():
            prediction = model(current_seq)
        last_word_vector = prediction # Get the last word vector
        match_word, max_cos = vector2word(last_word_vector.squeeze(0))
        print("predict=", match_word, max_cos)
        generated_words.append(match_word)

        # Update the sequence with the newly generated word
        # code here 
        if match_word in word_vector_dict:
            new_vector = word_vector_dict[match_word].unsqueeze(0).to(device)  # Ensure it has sequence and batch dimensions
            # Check if we need to append or remove the oldest word
            if current_seq.size(1) >= 20:  # If we already have 20 words
                current_seq = torch.cat((current_seq[:, 1:, :], new_vector.unsqueeze(1)), dim=1)  # Slide window
            else:
                current_seq = torch.cat((current_seq, new_vector.unsqueeze(1)), dim=1)  # Append new word
        else:
            print(f"Word '{match_word}' not found in dictionary, skipping.")
    return generated_words

# Load your pre-trained model
model_path = 'model_state_dict.pth'
model = load_model(model_path).to(device)
# with open('word_vector_dict.pkl', 'rb') as f:
#     word_vector_dict = pickle.load(f)
word_vector_dict = torch.load('word_vector_dict.pth')
# Prepare your initial sequence of 20 words
lines = "爹爹长叹一声：“本不想你进宫。只是事无可避也只得如此了。历代后宫都是是非之地况且今日云意殿选秀皇上已对你颇多关注想来今后必多是非一定要善自小心保全自己。”我忍着泪安慰爹爹："
lines = jieba.cut(lines)  # Your 20 initial words here
initial_words = []
for i in lines:
    initial_words.append(i)
    
initial_sequence = prepare_initial_sequence(initial_words, word_vector_dict).to(device)

# Generate words
generated_words = generate_words(model, initial_sequence, word_vector_dict)
print("Generated words:", generated_words)

