**Text Matching Model Architecture**

To solve the text matching task, we will use a Siamese Neural Network architecture with embeddings. The model will take two input texts and output a similarity score between them.

```python
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Define the dataset class
class TextMatchingDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text1 = self.df.iloc[idx, 0]
        text2 = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]

        encoding1 = self.tokenizer.encode_plus(
            text1,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        encoding2 = self.tokenizer.encode_plus(
            text2,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids1': encoding1['input_ids'].flatten(),
            'attention_mask1': encoding1['attention_mask'].flatten(),
            'input_ids2': encoding2['input_ids'].flatten(),
            'attention_mask2': encoding2['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define the model architecture
class TextMatchingModel(nn.Module):
    def __init__(self, model_name, hidden_size, dropout):
        super(TextMatchingModel, self).__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        outputs1 = self.bert(input_ids1, attention_mask=attention_mask1)
        outputs2 = self.bert(input_ids2, attention_mask=attention_mask2)

        pooled_output1 = outputs1.pooler_output
        pooled_output2 = outputs2.pooler_output

        diff = torch.abs(pooled_output1 - pooled_output2)
        prod = pooled_output1 * pooled_output2
        concat = torch.cat((diff, prod), dim=1)

        outputs = self.dropout_layer(concat)
        outputs = self.classifier(outputs)

        return outputs

# Load the dataset
df = pd.read_csv('data.csv')

# Preprocess the data
df = pd.get_dummies(df, columns=['main_genre', 'side_genre'])
df['text1'] = df.apply(lambda row: f"{row['Movie_Title']} {row['Director']}", axis=1)
df['text2'] = df.apply(lambda row: f"{row['Actors']} {row['Rating']}", axis=1)
df['label'] = df.apply(lambda row: 1 if row['Rating'] > 7 else 0, axis=1)

# Split the data into training and testing sets
train_text1, val_text1, train_text2, val_text2, train_labels, val_labels = train_test_split(df['text1'], df['text2'], df['label'], test_size=0.2, random_state=42)

# Create the dataset and data loader
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
dataset = TextMatchingDataset(df[['text1', 'text2', 'label']], tokenizer, max_len=512)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model, optimizer, and loss function
model = TextMatchingModel('distilbert-base-uncased', hidden_size=768, dropout=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# Train the model
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids1, attention_mask1, input_ids2, attention_mask2, labels = batch
        input_ids1, attention_mask1, input_ids2, attention_mask2, labels = input_ids1.to(device), attention_mask1.to(device), input_ids2.to(device), attention_mask2.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids1, attention_mask1, input_ids2, attention_mask2, batch_labels = batch
            input_ids1, attention_mask1, input_ids2, attention_mask2, batch_labels = input_ids1.to(device), attention_mask1.to(device), input_ids2.to(device), attention_mask2.to(device), batch_labels.to(device)

            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            logits = outputs.detach().cpu().numpy()
            batch_predictions = np.argmax(logits, axis=1)
            predictions.extend(batch_predictions)
            labels.extend(batch_labels.cpu().numpy())
    accuracy = accuracy_score(labels, predictions)
    print(f'Epoch {epoch+1}, Accuracy: {accuracy:.4f}')

# Hyperparameter tuning
param_grid = {
    'model_name': ['distilbert-base-uncased', 'bert-base-uncased'],
    'hidden_size': [256, 512, 768],
    'dropout': [0.1, 0.2, 0.3],
    'lr': [1e-5, 1e-4, 1e-3]
}

from sklearn.model_selection import GridSearchCV
from torch.optim import Adam

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(data_loader)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best accuracy: {grid_search.best_score_:.4f}')