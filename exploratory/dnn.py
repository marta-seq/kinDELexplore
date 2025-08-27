import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
import torch.nn as nn
import torch.optim as optim
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

class ChemBERTaDNNWrapper(nn.Module):
    def __init__(
            self,
            chemberta_model,
            tokenizer,
            dnn_layers=[128, 64],
            dropout=0.2,
            lr=1e-3,
            batch_size=32,  # Reduced from 64 for memory efficiency
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

        # Initialize mixed precision scaler
        self.scaler = GradScaler() if torch.cuda.is_available() else None

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
        device = next(self.chemberta_model.parameters()).device
        dummy_input = self.tokenizer(["CC"], return_tensors="pt", padding=True, truncation=True)
        dummy_input = {k: v.to(device) for k, v in dummy_input.items()}

        with torch.no_grad():
            embedding = self.chemberta_model(**dummy_input).last_hidden_state.mean(dim=1)
        return embedding.shape[1]

    def forward(self, smiles_list):
        # Ensure smiles_list is a list of strings
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        elif not isinstance(smiles_list, list):
            # Convert other types to list, ensuring they are strings
            smiles_list = [str(s) for s in smiles_list]

        # Validate input
        if not all(isinstance(s, str) for s in smiles_list):
            raise ValueError("All elements must be SMILES strings")

        device = next(self.chemberta_model.parameters()).device

        # Tokenize SMILES
        inputs = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Limit sequence length to save memory
        )
        # inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the same device as the model

        # Get embeddings - handle gradient computation properly
        if self.training and not self.freeze_chemberta:
            # During training with unfrozen ChemBERTa, allow gradients
            embeddings = self.chemberta_model(**inputs).last_hidden_state.mean(dim=1)

        else:
            # During evaluation or with frozen ChemBERTa
            with torch.no_grad():
                embeddings = self.chemberta_model(**inputs).last_hidden_state.mean(dim=1)
            # If training but ChemBERTa is frozen, detach and require gradients for DNN
            if self.training:
                embeddings = embeddings.detach().requires_grad_(True)
        embeddings = embeddings.to(device)  # Ensure embeddings are on the same device

        # Pass through DNN head
        return self.dnn_head(embeddings)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Validate inputs
        if not isinstance(X_train, list) or not all(isinstance(s, str) for s in X_train):
            raise ValueError("X_train must be a list of SMILES strings")

        if X_valid is not None and (not isinstance(X_valid, list) or not all(isinstance(s, str) for s in X_valid)):
            raise ValueError("X_valid must be a list of SMILES strings")

        # Convert labels to tensors on CPU first
        y_train_cpu = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        if y_valid is not None:
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).view(-1, 1).to(device)

        # Create DataLoader - keep data on CPU and move to GPU in batches
        train_dataset = TensorDataset(torch.arange(len(X_train)), y_train_cpu)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),  # Only use pin_memory if CUDA is available
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )

        criterion = nn.MSELoss()
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # Training loop
            self.train()
            total_train_loss = 0
            num_batches = 0

            for idx, yb in train_loader:
                # Move batch to device
                idx = idx.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                # Get SMILES batch
                smiles_batch = [X_train[i] for i in idx.cpu().numpy()]

                self.optimizer.zero_grad()

                # Use mixed precision if available
                if self.scaler is not None:
                    with autocast():
                        preds = self.forward(smiles_batch)
                        loss = criterion(preds, yb)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular training without mixed precision
                    preds = self.forward(smiles_batch)
                    loss = criterion(preds, yb)
                    loss.backward()
                    self.optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1

                # Clear cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_train_loss = total_train_loss / num_batches

            # Validation
            val_loss = None
            if X_valid is not None and y_valid is not None:
                self.eval()
                with torch.no_grad():
                    # Process validation in smaller chunks to avoid memory issues
                    # val_batch_size = min(32, len(X_valid))  # Use smaller batch size for validation
                    val_batch_size = self.batch_size  # Use same batch size as training
                    val_losses = []

                    for i in range(0, len(X_valid), val_batch_size):
                        end_idx = min(i + val_batch_size, len(X_valid))
                        X_batch = X_valid[i:end_idx]
                        y_batch = y_valid_tensor[i:end_idx]

                        preds = self.forward(X_batch)
                        batch_loss = criterion(preds, y_batch)
                        val_losses.append(batch_loss.item())

                        # Clear cache after each validation batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    val_loss = np.mean(val_losses)

                # LR scheduling
                if self.scheduler:
                    self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}  # Store on CPU
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        if best_state is not None:
                            self.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                        break

            # Verbose print
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss if val_loss else 'N/A'}"
            )

            # Clear cache at the end of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self

    def predict(self, X):
        if not isinstance(X, list) or not all(isinstance(s, str) for s in X):
            raise ValueError("X must be a list of SMILES strings")
        device = next(self.parameters()).device  # Get the device of the model

        self.eval()

        # Process predictions in batches to avoid memory issues
        predictions = []

        # Process in batches to avoid memory issues
        batch_size = self.batch_size
        for i in range(0, len(X), batch_size):
            end_idx = min(i + batch_size, len(X))
            X_batch = X[i:end_idx]

            with torch.no_grad():
                # Ensure input is moved to the same device as the model
                preds = self.forward(X_batch).to(device)
                predictions.extend(preds.cpu().numpy().flatten())
        return np.array(predictions)