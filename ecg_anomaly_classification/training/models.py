import torch
from torch import nn
from collections import OrderedDict


def create_mlp(layers_size, dropout):
    layers = [("layer{}".format(i),
               nn.Sequential(nn.Linear(layers_size[i - 1], layers_size[i]), nn.ReLU(), nn.Dropout(dropout)))
              for i in range(1, len(layers_size))]

    last_index = len(layers_size) - 1
    layers[-1] = (f"layer{last_index}", nn.Linear(layers_size[last_index - 1], layers_size[last_index]))

    return nn.Sequential(OrderedDict(layers))


class RNNBasedModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers, fc_layers):
        super(RNNBasedModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.fc_layers = fc_layers

        self.fc = create_mlp([hidden_size] + fc_layers + [1], dropout)

    def forward(self, x):
        x, _ = self.rnn_block(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

    def extended_state_dict(self):
        return {
            "type": self.rnn_type,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "num_layers": self.num_layers,
            "fc_layers": self.fc_layers,
            "weights": self.state_dict()
        }


class LSTM(RNNBasedModel):
    def __init__(self, input_size, hidden_size, dropout, num_layers, fc_layers):
        super(LSTM, self).__init__(input_size, hidden_size, dropout, num_layers, fc_layers)

        self.rnn_block = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.rnn_type = "lstm"


class GRU(RNNBasedModel):
    def __init__(self, input_size, hidden_size, dropout, num_layers, fc_layers):
        super(GRU, self).__init__(input_size, hidden_size, dropout, num_layers, fc_layers)

        self.rnn_block = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.rnn_type = "gru"


class CNN(nn.Module):
    def __init__(self, input_size, cnn_layers, dropout, fc_layers):
        super(CNN, self).__init__()

        self.input_size = input_size
        self.cnn_layers = cnn_layers
        self.dropout = dropout
        self.fc_layers = fc_layers

        num_layers = len(cnn_layers)
        self.cnn_block = []

        prev_filters = input_size
        for i in range(num_layers - 1):
            cur_filters = cnn_layers[i][0]
            self.cnn_block.append((f"layers{i}", nn.Sequential(nn.Conv1d(prev_filters, cur_filters, cnn_layers[i][1]),
                                                               nn.ReLU(),
                                                               nn.MaxPool1d(2),
                                                               nn.BatchNorm1d(cur_filters))))
            prev_filters = cur_filters

        self.cnn_block.append((f"layers{num_layers - 1}", nn.Sequential(nn.Conv1d(prev_filters,
                                                                                  cnn_layers[num_layers - 1][0],
                                                                                  cnn_layers[num_layers - 1][1]),
                                                                        nn.ReLU())))
        self.cnn_block = nn.Sequential(OrderedDict(self.cnn_block))
        self.fc = create_mlp([cnn_layers[-1][0]] + fc_layers + [1], dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn_block(x)
        x = x.mean(dim=-1)
        x = self.fc(x)
        return x

    def extended_state_dict(self):
        return {
            "type": "cnn",
            "input_size": self.input_size,
            "cnn_layers": self.cnn_layers,
            "dropout": self.dropout,
            "fc_layers": self.fc_layers,
            "weights": self.state_dict()
        }


def get_model(config, input_size):
    config = config.copy()
    model_type = config["type"]
    del config["type"]

    if model_type == "cnn":
        return CNN(input_size=input_size, **config)
    elif model_type == "gru":
        return GRU(input_size=input_size, **config)
    elif model_type == "lstm":
        return LSTM(input_size=input_size, **config)

    raise TypeError(f"Unknown model type: {model_type}")


def load_model(model_path):
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model_type, weights = state_dict["type"], state_dict["weights"]
    del state_dict["type"], state_dict["weights"]

    if model_type == "cnn":
        model = CNN(**state_dict)
    elif model_type == "gru":
        model = GRU(**state_dict)
    elif model_type == "lstm":
        model = LSTM(**state_dict)
    else:
        raise TypeError(f"Unknown model type: {model_type}")

    model.load_state_dict(weights)
    return model
