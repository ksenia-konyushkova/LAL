import copy

import torch
from numpy import ndarray
from sklearn import clone
from torch import nn, Tensor, optim
import numpy as np
from torch.nn import CrossEntropyLoss


class Model:
    """
    An interface for an ML model that mimics that of sklearn. The code was originally written to be used exclusively for
    RandomForestClassifiers, but we wanted to try out other model classes, so we introduced this quick and dirty
    interface to prevent having to change the existing code too much.
    """
    def fit(self, X: ndarray, y: ndarray):
        """
        Fit the model on the given data and return self. Both inputs should be 2-dimensional:

        X: num_examples x input_dim
        y: num_examples x 1
        """
        raise NotImplementedError

    def predict(self, X: ndarray) -> ndarray:
        """
        Use the current state of the model's parameters to predict class labels
        for all examples in X

        X: num_examples x input_dim

        Output: num_examples x 1
        """
        raise NotImplementedError

    def predict_proba(self, X: ndarray):
        """
        Use the current state of the model's parameters to make class probability
        predictions for each example in X

        X: num_examples x input_dim

        Output: num_examples x output_classes
        """
        raise NotImplementedError

    def clone(self):
        """
        Return a copy of the model with the same parameters, not fitted on data yet
        """
        raise NotImplementedError


class SKLearnModel(Model):
    def __init__(self, model):
        self.model = model

    def fit(self, X: ndarray, y: ndarray):
        # sklearn wants the labels in a 1-d array
        return self.model.fit(X, y.flatten())

    def predict(self, X) -> ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: ndarray) -> ndarray:
        return self.model.predict_proba(X)

    def clone(self):
        return clone(self.model)


class PyTorchModel(Model):
    def __init__(
            self,
            model: nn.Module,
            epochs,
            lr,
            lr_decay=0.99,
            early_stopping_patience=None
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.early_stopping_patience = early_stopping_patience
        self.criterion = CrossEntropyLoss()
        self.optimizer = optim.Adam(
            list(model.parameters()), lr=lr
        )

        self.total_params = 0
        for p in self.model.parameters():
            self.total_params += self._shape_size(p.data.shape)

    def set_parameters(self, new_parameters: Tensor):
        assert len(new_parameters.shape) == 1
        assert new_parameters.shape[0] == self.total_params
        offset = 0
        for param in self.model.parameters():
            shape_size = self._shape_size(param.data.shape)
            param.data = new_parameters[offset:offset + shape_size].reshape(param.data.shape)
            offset += shape_size

    def _shape_size(self, size: torch.Size):
        mult = 1
        for dim in size:
            mult *= dim
        return mult

    def fit(self, X: ndarray, y: ndarray):
        best_loss = float("inf")
        early_stop = 0

        train_data = torch.from_numpy(X).float()
        train_labels = torch.from_numpy(y).long()
        train_labels = train_labels.view(-1)

        for epoch in range(self.epochs):

            self.optimizer.param_groups[0]["lr"] *= self.lr_decay

            outputs = self.model.forward(train_data)
            train_loss = self.criterion(outputs, train_labels)

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            if self.early_stopping_patience:
                if train_loss.item() < best_loss:
                    early_stop = 0
                    best_loss = train_loss.item()
                else:
                    early_stop += 1

                if early_stop > self.early_stopping_patience:
                    return self

        return self

    def predict(self, X) -> ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: ndarray) -> ndarray:
        with torch.no_grad():
            return self.model.forward(torch.from_numpy(X).float()).numpy()

    def clone(self):
        return PyTorchModel(
            copy.deepcopy(self.model),
            self.epochs,
            self.lr,
            self.lr_decay,
            self.early_stopping_patience
        )


class ACNMLModel(Model):
    pass


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)