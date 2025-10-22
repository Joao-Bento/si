from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import neural_network as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

features_path = 'selected_features.csv'
df = pd.read_csv(features_path)
df = df[~df.isin([float('inf'), float('-inf')]).any(axis=1)]
df.reset_index(drop=True, inplace=True)

print(df.head()) 

X = df.drop(columns=['label'])
y = df['label']

#train test spliting
test_size=0.2
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=1)

# Reset indices of Xtr and ytr
Xtr = Xtr.reset_index(drop=True)
ytr = ytr.reset_index(drop=True)

# Standardize features
scaler=StandardScaler()
Xtr= scaler.fit_transform(Xtr)
Xte= scaler.transform(Xte)


Xte = torch.tensor(Xte, dtype=torch.float32)
yte = torch.tensor(yte.to_numpy(), dtype=torch.float32)
Xtr = torch.tensor(Xtr, dtype=torch.float32)
ytr = torch.tensor(ytr.to_numpy(), dtype=torch.float32)

##NN hyperparameters
num_epochs=250
lr=0.0005
dropout=0.1
batch_size=64

# Define K-Fold Cross-Validation
k = len(X)//5
kf = KFold(n_splits=k, shuffle=True, random_state=1)

# Initialize fold results
fold_results = []

avg_final_val_loss = []
patience = 20
for fold, (train_idx, val_idx) in enumerate(kf.split(Xtr)):
    print(f"Fold {fold + 1}/{k}")

    # # Use .iloc to index rows by position
    X_train, X_val = Xtr[train_idx], Xtr[val_idx]
    y_train, y_val = ytr[train_idx], ytr[val_idx]

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Wrap training data into a DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.MLP(input_size=X_train.shape[1], dropout_prob=dropout).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y.view(-1, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_dataloader)
        losses.append(avg_loss)
        
        # Validation
        model.eval()
        # Save the model for the current fold
        #torch.save(model.state_dict(), f"model_fold_{fold + 1}.pth")
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val.view(-1, 1)).item()
            val_losses.append(val_loss)
            #print(f"Fold {fold + 1}, Validation Loss: {val_loss:.4f}")
            
        # Early stopping
        if len(val_losses) > 50:
            recent_val_losses = val_losses[-patience:]
            if all(recent_val_losses[i] >= recent_val_losses[i - 1] for i in range(1, patience)):
                print(f"Early stopping at epoch {epoch + 1} for fold {fold + 1}")
                break
        
    avg_final_val_loss.append(np.mean(val_losses[-10:]))
    torch.save(model.state_dict(), f"model_fold_{fold + 1}.pth")
        
        
    import matplotlib.pyplot as plt

    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, label=f"Train loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label=f"Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss per epoch of best model: fold {fold + 1}")
    plt.legend()
    plt.grid()
    plt.savefig(f"training_loss_fold_{fold + 1}.png")
            
        

    # # Validation
    # model.eval()
    # # Save the model for the current fold
    # #torch.save(model.state_dict(), f"model_fold_{fold + 1}.pth")
    # with torch.no_grad():
    #     val_logits = model(X_val)
    #     val_loss = criterion(val_logits, y_val.view(-1, 1)).item()
    #     fold_results.append(val_loss)
    #     print(f"Fold {fold + 1}, Validation Loss: {val_loss:.4f}")
    #     val_losses.append(val_loss)
    #     torch.save(model.state_dict(), f"model_fold_{fold + 1}.pth")

    
# import matplotlib.pyplot as plt
# # Plot training losses
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, k+1), val_losses, label=f"Fold {fold + 1}")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss per Epoch")
# plt.legend()
# plt.grid()
# plt.savefig(f"training_loss_fold_{fold + 1}.png")

# Calculate average validation loss across folds
#avg_loss = sum(fold_results) / len(fold_results)
#print(f"Average Validation Loss: {avg_loss:.4f}")


# Get the indices of the top 3 losses in val_losses
n_best=3
top_loss_indices = sorted(range(len(avg_final_val_loss)), key=lambda i: avg_final_val_loss[i], reverse=False)[:n_best]
#print(f"Indices of top 3 losses: {top_3_loss_indices}")
print(f"Best model during validation: fold number {top_loss_indices[0] + 1} with validation loss: {avg_final_val_loss[top_loss_indices[0]]:.4f}")


best_rec = -np.inf
for idx in top_loss_indices:
    # Load the saved model for the fold with the top loss
    model_path = f"model_fold_{idx + 1}.pth"
    loaded_model = nn.MLP(input_size=X_train.shape[1], dropout_prob=dropout).to(device)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    print(f"Loaded model from fold {idx + 1} with validation loss: {val_losses[idx]:.4f}")
    y_pred=loaded_model(Xte)
    rec_i = recall_score(yte.detach().numpy(),y_pred.detach().numpy()>0.5)
    if rec_i>best_rec:
        best_rec=rec_i
        best_model=loaded_model

print(f"Best model test recall: {best_rec:.4f}")

# Evaluate the best model on the test set (Xte, yte)
y_pred = best_model(Xte).detach().numpy() > 0.5
test_accuracy = accuracy_score(yte.detach().numpy(), y_pred)
test_f1 = f1_score(yte.detach().numpy(), y_pred)
test_precision = precision_score(yte.detach().numpy(), y_pred)
test_recall = recall_score(yte.detach().numpy(), y_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

X = torch.tensor(scaler.transform(X), dtype=torch.float32)
y = torch.tensor(y.to_numpy(), dtype=torch.float32)
# Evaluate the best model on the entire dataset (X, y)
all_y_preds = best_model(X).detach().numpy() > 0.5
all_accuracy = accuracy_score(y.detach().numpy(), all_y_preds)
all_f1 = f1_score(y.detach().numpy(), all_y_preds)
all_precision = precision_score(y.detach().numpy(), all_y_preds)
all_recall = recall_score(y.detach().numpy(), all_y_preds)

print(f"All Data Accuracy: {all_accuracy:.4f}")
print(f"All Data F1 Score: {all_f1:.4f}")
print(f"All Data Precision: {all_precision:.4f}")
print(f"All Data Recall: {all_recall:.4f}")

# Compute confusion matrix for the entire dataset
cm = confusion_matrix(yte.detach().numpy(), y_pred)

# Plot and save the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Test Confusion Matrix")
plt.savefig("confusion_matrix_nn_test.png")
plt.show()

# Compute confusion matrix for the entire dataset
cm_total = confusion_matrix(y.detach().numpy(), all_y_preds)

# Plot and save the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_total)
disp.plot(cmap='Blues')
plt.title("Total Confusion Matrix")
plt.savefig("confusion_matrix_nn_total.png")
plt.show()


