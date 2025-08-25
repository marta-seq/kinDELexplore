import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout_prob=0.2):
        super(DNNModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_prob),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_prob),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_prob),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_prob),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_prob),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1)  # Output layer (no activation for regression)
        )

    def forward(self, x):
        return self.layers(x)



class DNNWrapper:
    def __init__(self, input_dim, layers=[128, 64], dropout=0.2, lr=1e-3,
                 batch_size=64, epochs=50, optimizer="adam",
                 weight_decay=0.0, patience=10,
                 lr_scheduler=True, lr_factor=0.5, lr_patience=5):
        import torch.nn as nn
        import torch.optim as optim

        self.input_dim = input_dim
        self.layers = layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.patience = patience
        self.lr_scheduler_flag = lr_scheduler
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

        # Build model
        modules = []
        in_dim = input_dim
        for h in layers:
            modules.append(nn.Linear(in_dim, h))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            in_dim = h
        modules.append(nn.Linear(in_dim, 1))  # regression output

        self.model = nn.Sequential(*modules)

        # Optimizer with weight decay
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Scheduler (ReduceLROnPlateau)
        self.scheduler = None
        if lr_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=lr_factor,
                patience=lr_patience,
                verbose=True
            )

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Convert data
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=self.batch_size, shuffle=True)

        if X_valid is not None and y_valid is not None:
            X_valid = torch.tensor(X_valid, dtype=torch.float32)
            y_valid = torch.tensor(y_valid, dtype=torch.float32).view(-1, 1)

        criterion = nn.MSELoss()
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training loop
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                self.optimizer.step()

            # Validation
            val_loss = None
            if X_valid is not None:
                self.model.eval()
                with torch.no_grad():
                    preds = self.model(X_valid.to(device))
                    val_loss = criterion(preds, y_valid.to(device)).item()

                # LR scheduling
                if self.scheduler:
                    self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        self.model.load_state_dict(best_state)
                        break
            # Verbose print
            train_loss_val = loss.item() if torch.is_tensor(loss) else loss
            val_loss_val = val_loss.item() if torch.is_tensor(val_loss) else val_loss

            logger.info(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss_val:.4f}, Val Loss: {val_loss_val:.4f}")

        return self

    def predict(self, X):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model(X).cpu().numpy().flatten()
        return preds


#
#
# class ChemBERTaDNNWrapper(nn.Module):
#     def __init__(
#         self,
#         chemberta_model,
#         dnn_layers=[128, 64],
#         dropout=0.2,
#         lr=1e-3,
#         batch_size=64,
#         epochs=50,
#         optimizer="adam",
#         weight_decay=0.0,
#         patience=10,
#         lr_scheduler=True,
#         lr_factor=0.5,
#         lr_patience=5,
#         freeze_chemberta=False,  # Set to False for fine-tuning
#     ):
#         super(ChemBERTaDNNWrapper, self).__init__()
#         self.chemberta_model = chemberta_model
#         self.dnn_layers = dnn_layers
#         self.dropout = dropout
#         self.lr = lr
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.optimizer_name = optimizer
#         self.weight_decay = weight_decay
#         self.patience = patience
#         self.lr_scheduler_flag = lr_scheduler
#         self.lr_factor = lr_factor
#         self.lr_patience = lr_patience
#         self.freeze_chemberta = freeze_chemberta
#
#         # Freeze or unfreeze ChemBERTa layers
#         for param in self.chemberta_model.parameters():
#             param.requires_grad = not freeze_chemberta
#
#         # Get the embedding dimension from ChemBERTa
#         embedding_dim = self._get_embedding_dim()
#
#         # Build DNN head
#         modules = []
#         in_dim = embedding_dim
#         for h in dnn_layers:
#             modules.append(nn.Linear(in_dim, h))
#             modules.append(nn.ReLU())
#             modules.append(nn.Dropout(dropout))
#             in_dim = h
#         modules.append(nn.Linear(in_dim, 1))  # regression output
#         self.dnn_head = nn.Sequential(*modules)
#
#         # Optimizer with weight decay
#         if optimizer == "adam":
#             self.optimizer = optim.Adam(
#                 [
#                     {"params": self.chemberta_model.parameters(), "lr": lr / 10},  # Lower learning rate for ChemBERTa
#                     {"params": self.dnn_head.parameters(), "lr": lr},
#                 ],
#                 weight_decay=weight_decay,
#             )
#         elif optimizer == "sgd":
#             self.optimizer = optim.SGD(
#                 [
#                     {"params": self.chemberta_model.parameters(), "lr": lr / 10},
#                     {"params": self.dnn_head.parameters(), "lr": lr},
#                 ],
#                 weight_decay=weight_decay,
#             )
#         else:
#             raise ValueError(f"Unknown optimizer: {optimizer}")
#
#         # Scheduler (ReduceLROnPlateau)
#         self.scheduler = None
#         if lr_scheduler:
#             self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#                 self.optimizer,
#                 mode="min",
#                 factor=lr_factor,
#                 patience=lr_patience,
#                 verbose=True,
#             )
#
#     def _get_embedding_dim(self):
#         # Dummy forward pass to get embedding dimension
#         dummy_input = torch.randn(1, 3, 224, 224)  # Adjust shape as needed
#         with torch.no_grad():
#             embedding = self.chemberta_model(dummy_input)
#         return embedding.shape[1]
#
#     def forward(self, x):
#         embedding = self.chemberta_model(x)
#         return self.dnn_head(embedding)
#
#     def fit(self, X_train, y_train, X_valid=None, y_valid=None):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.to(device)
#
#         # Convert data
#         X_train = torch.tensor(X_train, dtype=torch.float32)
#         y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
#         train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
#
#         if X_valid is not None and y_valid is not None:
#             X_valid = torch.tensor(X_valid, dtype=torch.float32)
#             y_valid = torch.tensor(y_valid, dtype=torch.float32).view(-1, 1)
#
#         criterion = nn.MSELoss()
#         best_val_loss = float("inf")
#         patience_counter = 0
#
#         for epoch in range(self.epochs):
#             # Training loop
#             self.train()
#             for xb, yb in train_loader:
#                 xb, yb = xb.to(device), yb.to(device)
#                 self.optimizer.zero_grad()
#                 preds = self(xb)
#                 loss = criterion(preds, yb)
#                 loss.backward()
#                 self.optimizer.step()
#
#             # Validation
#             val_loss = None
#             if X_valid is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     preds = self(X_valid.to(device))
#                     val_loss = criterion(preds, y_valid.to(device)).item()
#
#                 # LR scheduling
#                 if self.scheduler:
#                     self.scheduler.step(val_loss)
#
#                 # Early stopping
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     patience_counter = 0
#                     best_state = self.state_dict()
#                 else:
#                     patience_counter += 1
#                     if patience_counter >= self.patience:
#                         logger.info(f"Early stopping at epoch {epoch + 1}")
#                         self.load_state_dict(best_state)
#                         break
#
#             # Verbose print
#             logger.info(
#                 f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss if val_loss else 'N/A'}"
#             )
#
#         return self
#
#     def predict(self, X):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.eval()
#         X = torch.tensor(X, dtype=torch.float32).to(device)
#         with torch.no_grad():
#             preds = self(X).cpu().numpy().flatten()
#         return preds
class ChemBERTaDNNWrapper(nn.Module):
    def __init__(
        self,
        chemberta_model,
        tokenizer,
        dnn_layers=[128, 64],
        dropout=0.2,
        lr=1e-3,
        batch_size=64,
        epochs=50,
        optimizer="adam",
        weight_decay=0.0,
        patience=10,
        lr_scheduler=True,
        lr_factor=0.5,
        lr_patience=5,
        freeze_chemberta=False,
    ):
        super(ChemBERTaDNNWrapper, self).__init__()
        self.chemberta_model = chemberta_model
        self.tokenizer = tokenizer
        self.dnn_layers = dnn_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.patience = patience
        self.lr_scheduler_flag = lr_scheduler
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.freeze_chemberta = freeze_chemberta

        # Freeze or unfreeze ChemBERTa layers
        for param in self.chemberta_model.parameters():
            param.requires_grad = not freeze_chemberta

        # Get the embedding dimension from ChemBERTa
        embedding_dim = self._get_embedding_dim()

        # Build DNN head
        modules = []
        in_dim = embedding_dim
        for h in dnn_layers:
            modules.append(nn.Linear(in_dim, h))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            in_dim = h
        modules.append(nn.Linear(in_dim, 1))  # Regression output
        self.dnn_head = nn.Sequential(*modules)

        # Optimizer with weight decay
        if optimizer == "adam":
            self.optimizer = optim.Adam(
                [
                    {"params": self.chemberta_model.parameters(), "lr": lr / 10},
                    {"params": self.dnn_head.parameters(), "lr": lr},
                ],
                weight_decay=weight_decay,
            )
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(
                [
                    {"params": self.chemberta_model.parameters(), "lr": lr / 10},
                    {"params": self.dnn_head.parameters(), "lr": lr},
                ],
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Scheduler (ReduceLROnPlateau)
        self.scheduler = None
        if lr_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=lr_factor,
                patience=lr_patience,
            )

    def _get_embedding_dim(self):
        # Dummy input to get embedding dimension
        dummy_input = self.tokenizer(["CC"], return_tensors="pt", padding=True, truncation=True).to(self.chemberta_model.device)
        with torch.no_grad():
            embedding = self.chemberta_model(**dummy_input).last_hidden_state.mean(dim=1)
        return embedding.shape[1]

    def forward(self, smiles_list):
        # Ensure smiles_list is a list of strings
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        elif isinstance(smiles_list, torch.Tensor) or isinstance(smiles_list, np.ndarray):
            # If it's a tensor or array, assume it's indices and fetch the corresponding SMILES
            smiles_list = [str(s) for s in smiles_list]
        # Tokenize SMILES
        inputs = self.tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True).to(self.chemberta_model.device)
        # Get embeddings
        with torch.no_grad():
            embeddings = self.chemberta_model(**inputs).last_hidden_state.mean(dim=1)
        # Pass through DNN head
        return self.dnn_head(embeddings)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Convert labels to tensors
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        if y_valid is not None:
            y_valid = torch.tensor(y_valid, dtype=torch.float32).view(-1, 1).to(device)

        # Create DataLoader for training data
        train_dataset = TensorDataset(torch.arange(len(X_train)), y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training loop
            self.train()
            for idx, yb in train_loader:
                idx, yb = idx.to(device), yb.to(device)
                smiles_batch = [X_train[i] for i in idx.cpu().numpy()]
                self.optimizer.zero_grad()
                preds = self.forward(smiles_batch)
                loss = criterion(preds, yb)
                loss.backward()
                self.optimizer.step()

            # Validation
            val_loss = None
            if X_valid is not None and y_valid is not None:
                self.eval()
                with torch.no_grad():
                    # Ensure X_valid is a list of SMILES strings
                    if not isinstance(X_valid, list):
                        raise ValueError(f"X_valid must be a list of SMILES strings. Got {type(X_valid)} instead.")

                    preds = self.forward(X_valid)
                    val_loss = criterion(preds, y_valid).item()

                # LR scheduling
                if self.scheduler:
                    self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        self.load_state_dict(best_state)
                        break

            # Verbose print
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss if val_loss else 'N/A'}"
            )

        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            preds = self.forward(X).cpu().numpy().flatten()
        return preds

