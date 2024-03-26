import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from random import randint
from gensim.models import Word2Vec
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
seq = []
max_w = 50
float_size = 4
word_vector_dict = {}

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.fc(lstm_out[:, -1, :])
        return y_pred

def load_vectors(model_path):
    print("Begin loading model")
    
    # 加载模型
    model = Word2Vec.load(model_path)
    print(f"Model loaded with {len(model.wv)} words and vector size = {model.vector_size}")
 
    # 遍历模型中的每个词，将其添加到字典中
    for word in model.wv.index_to_key:
        # 将gensim的numpy向量转换为torch张量
        vector = torch.tensor(model.wv[word], dtype=torch.float)
        word_vector_dict[word] = vector
    
    print("Model vectors loaded successfully")
    return word_vector_dict

def init_seq(sequence_length=20):
    file_object = open('novel.segment', 'r', encoding='utf-8')
    temp_seq = []
    while True:
        line = file_object.readline()
        if line:
            for word in line.split(' '):
                if word in word_vector_dict:
                    temp_seq.append(word_vector_dict[word])
                    if len(temp_seq) >= sequence_length + 1:  # +1 for the label
                        # Take the last sequence_length+1 words
                        seq.append(temp_seq[-(sequence_length+1):])
                        temp_seq.pop(0)  # Remove the first element to slide the window
        else:
            break
    file_object.close()

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

def main():
    load_vectors("./novel.bin")
    torch.save(word_vector_dict,"word_vector_dict.pth")
    try:
        with open('sequences.pkl', 'rb') as f:
            sequences = pickle.load(f)
        with open('labels.pkl', 'rb') as f:
            labels = pickle.load(f)
        print("Pickle files founded. Generating sequences and labels Sucessfully.")
    except FileNotFoundError:
        # 初始化序列
        init_seq(sequence_length=20)  # 使用修改过的函数，确保每个序列长度为20
        # 将seq列表中的数据转换为Tensor
        sequences = torch.stack([torch.stack(item[:-1]) for item in seq]).to(device)  # Using the first elements as input
        labels = torch.stack([item[-1] for item in seq]).to(device)  # Using the last element as label
        with open('sequences.pkl', 'wb') as f:
            pickle.dump(sequences, f)
        with open('labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
    
    # 数据集划分
    train_size = int(len(sequences) * 0.8)
    train_sequences, val_sequences = sequences[:train_size], sequences[train_size:]
    train_labels, val_labels = labels[:train_size], labels[train_size:]
    
    # 创建TensorDataset和DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_sequences, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_sequences, val_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)

    # 模型初始化
    model = LSTMModel(200, 200, 200).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(500):
        start_time = time.time()
        model.train()
        total_loss = 0
        # Initialize tqdm loop with a description and total number of batches
        train_loader_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/500 Training', leave=False)
        for inputs, targets in train_loader_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Update the progress bar with the latest loss
            train_loader_bar.set_postfix(loss=f'{loss.item():.4f}', refresh=True)
            
        avg_loss = total_loss / len(train_loader)
        
        # Similar logic can be applied to the validation loop if desired
        model.eval()
        total_val_loss = 0
        val_loader_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/500 Validating', leave=False)
        with torch.no_grad():
            for inputs, targets in val_loader_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                total_val_loss += val_loss.item()
                val_loader_bar.set_postfix(val_loss=f'{val_loss.item():.4f}', refresh=True)

        avg_val_loss = total_val_loss / len(val_loader)
        end_time = time.time()

        print(f'Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {end_time - start_time:.2f}s')
        
        test_index = randint(0, len(val_sequences))  # Ensure there's enough room for 20 words + label
        test_seq = val_sequences[test_index]
        test_label = val_labels[test_index] # Next words as labels
        test_seq = test_seq.unsqueeze(0) if test_seq.dim() == 2 else test_seq
        # Testing phase
        model.eval()
        with torch.no_grad():
            test_output = model(test_seq)
            print(test_output)
            test_loss = criterion(test_output, test_label)
            p_word, _ = vector2word(test_output.squeeze(0))
            real_word,_ = vector2word(test_label.squeeze(0))
            # Convert output to words...
            print(f"Epoch {epoch+1}, Test Loss: {test_loss.item():.4f}")
            
            # Converting the output vector back to word (you might need to adjust the logic based on your vector2word function)
           
            
            print(f"Test Output Words: {p_word}")
            print(f"Real Words: {real_word}")
            print(f"Test Loss: {test_loss.item():.4f}")
    # Your code to save the model...   
    torch.save(model.state_dict(), 'model_state_dict.pth')
if __name__ == "__main__":
    main()
