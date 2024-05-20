import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape,
        num_classes,
        hidden_sizes=(10, 10),
        activation="relu",
        bias=False,
        dropout=False,
        residual=False,
        save=False,
        batch_norm=False
    ):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.activation = activation
        self.input_shape = input_shape
        ch, w, h = self.input_shape
        self.input_size = ch * w * h
        
        self.hidden_sizes = hidden_sizes
        self.layers_size = list(hidden_sizes)
        self.layers_size.insert(0, ch * w * h)
        self.layers_size.append(num_classes)
        self.num_layers = len(self.layers_size) + 2
        
        self.residual = residual
        self.save = save
        self.bias = bias
        self.batch_norm = batch_norm
        self.count = 0

        self.layers = nn.ModuleList()
        
        self.matrix_input_dim = self.input_size + 1 if bias or batch_norm else self.input_size
        
        for idx in range(len(self.layers_size) - 1):
            self.layers.append(nn.Linear(self.layers_size[idx],
                                         self.layers_size[idx + 1],
                                         bias=bias))

            if idx == len(self.layers_size) - 2:
                break

            if batch_norm:
                self.layers.append(nn.BatchNorm1d(self.layers_size[idx + 1]))

            if dropout and idx % 2:
                self.layers.append(nn.Dropout(0.5))

            self.layers.append(self.get_activation_fn()())

    def forward(self, x, rep=False):
        x = x.view(-1, self.input_size)

        if not rep:
            for layer in self.layers:
                x = layer(x)

            return x

        self.pre_acts = []
        self.acts = []

        x = self.layers[0](x) # Linear

        if self.save:
            self.pre_acts.append(x.clone().detach())

        x = self.layers[1](x) # RELU

        if self.save:
            self.acts.append(x.clone().detach())

        x_res = x

        cont = 0

        for i in range(2, len(self.layers)):
            layer = self.layers[i]

            if isinstance(layer, nn.Linear):
                x = layer(x)
                cont += 1

                if self.save:
                    self.pre_acts.append(x.clone().detach())

            elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.ELU, nn.LeakyReLU, nn.PReLU, nn.Sigmoid)):
                if self.residual:
                    if cont % 4 == 0:  # Add the saved input after a residual pair
                        x = x + x_res
                        x = layer(x)
                        x_res = x
                    else:
                        x = layer(x)

                else:
                    x = layer(x)

                if self.save:
                    self.acts.append(x.clone().detach())
        return x

    def get_weights(self):
        w = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                w.append(m.weight.data)
        return w

    def get_biases(self):
        b = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                b.append(m.bias.data)
        return b

    def get_activation_fn(self):
        act_name = self.activation.lower()
        activation_fn_map = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "leakyrelu": nn.LeakyReLU,
            "prelu": nn.PReLU,
            "sigmoid": nn.Sigmoid,
        }
        if act_name not in activation_fn_map.keys():
            raise ValueError("Unknown activation function name : ")
        return activation_fn_map[act_name]

    def init(self):
        def init_func(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.running_mean = torch.randn(200)
                m.running_var = torch.rand(200)
                m.reset_parameters()

        self.apply(init_func)
