### Sentiment Analysis Model

We will use a combination of natural language processing (NLP) and machine learning techniques to build a sentiment analysis model.

#### Model Architecture

We will use a transformer-based architecture, specifically the BERT model, which has achieved state-of-the-art results in various NLP tasks.

```python
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Define a custom dataset class for our data
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Add a classification layer on top of BERT
class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return outputs
```

#### Training Pipeline

```python
# Set hyperparameters
MAX_LEN = 512
BATCH_SIZE = 32
EPOCHS = 5

# Load data and split into training and validation sets
df = pd.read_csv("data.csv")
X = df["text"]
y = df["sentiment"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create datasets and data loaders
train_dataset = SentimentDataset(X_train, y_train, tokenizer, MAX_LEN)
val_dataset = SentimentDataset(X_val, y_val, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model, optimizer, and loss function
model = SentimentClassifier()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# Train model
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

    model.eval()
    with torch.no_grad():
        total_correct = 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, dim=1)

            total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / len(val_dataset)
        print(f"Epoch {epoch+1}, Val Accuracy: {accuracy:.4f}")
```

#### Hyperparameter Tuning

We will use the `optuna` library to perform hyperparameter tuning.

```python
import optuna

# Define hyperparameter search space
def objective(trial):
    # Hyperparameters to tune
    MAX_LEN = trial.suggest_categorical("max_len", [128, 256, 512])
    BATCH_SIZE = trial.suggest_categorical("batch_size", [16, 32, 64])
    EPOCHS = trial.suggest_int("epochs", 1, 10)
    LR = trial.suggest_loguniform("lr", 1e-6, 1e-3)

    # Initialize model, optimizer, and loss function
    model = SentimentClassifier()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # Train model
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

        model.eval()
        with torch.no_grad():
            total_correct = 0
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, dim=1)

                total_correct += (predicted == labels).sum().item()

            accuracy = total_correct / len(val_dataset)
            print(f"Epoch {epoch+1}, Val Accuracy: {accuracy:.4f}")

    return accuracy

# Perform hyperparameter search
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)
print("Best accuracy:", study.best_value)
```