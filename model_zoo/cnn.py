import torch
import torch.nn as nn
import math


class CNN_2D(nn.Module):
    """
    Implementation for (almost) arbitrary CNN.
    This class has a forward method that can be used for training or to 
    save activations and preactivations to later compute the matrix.

    Args:
        input_shape (tuple[int,int,int]): Input shape as (channels, height, width)
        num_classes (int): Number of output classes
        channels (tuple[int]): Number of channels for each conv layer
        padding (tuple[(int,int)]): Padding for each conv layer
        fc (tuple[int]): Number of neurons for each linear layer
        kernel_size (tuple[(int,int)]): Kernel size for each conv layer
        bias (bool): Whether to use bias in the linear layers
        residual (list[(int,int)]): List of layers to add residual connections between
        batch_norm (bool): Whether to use batch normalization
        dropout (bool): Whether to use dropout
        activation (str): Activation function to use
        pooling (str): Pooling function to use
        save (bool): Whether to save activations
    """

    def __init__(
            self,
            input_shape,
            num_classes: int,
            channels=(8, 16),
            padding=((1, 1), (1, 1)),
            fc=(500),
            kernel_size=((3, 3), (3, 3)),
            bias: bool = False,
            residual=[],
            batch_norm: bool = False,
            dropout: bool = False,
            activation: str = "relu",
            pooling: str = "avg",
            save: bool = False
    ) -> None:
        super().__init__()
        c, h, w = input_shape
        if len(kernel_size) != len(channels):
            kernel_size = [kernel_size[0] for _ in channels]
        if len(padding) != len(channels):
            padding = [padding[0] for _ in channels]

        # Correctly compute shape after each conv layer
        shape_per_layer = []
        current_h = h
        current_w = w
        for i in range(len(channels)):
            out_h = (current_h + 2 * padding[i][0] - kernel_size[i][0]) // 1 + 1
            out_w = (current_w + 2 * padding[i][1] - kernel_size[i][1]) // 1 + 1
            shape_per_layer.append((channels[i], out_h, out_w))
            current_h = out_h
            current_w = out_w

        self.input_shape = input_shape
        self.save = save
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.residual = {
            a: b for a, b in list(set(residual))
            if a < b and shape_per_layer[a] == shape_per_layer[b]
        }
        self.bias = bias
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.matrix_input_dim = c * w * h + 1 if bias or batch_norm else c * w * h
        self.activation = activation
        self.pooling = pooling

        self.conv_layers.append(
            nn.Conv2d(
                in_channels=c,
                out_channels=channels[0],
                kernel_size=kernel_size[0],
                padding=padding[0],
                bias=False
            )
        )

        if batch_norm:
            self.conv_layers.append(
                nn.BatchNorm2d(channels[0])
            )

        if isinstance(channels, tuple):
            for i in range(1, len(channels)):
                self.conv_layers.append(self.get_activation_fn())
                self.conv_layers.append(
                    nn.Conv2d(
                        in_channels=channels[i - 1],
                        out_channels=channels[i],
                        kernel_size=kernel_size[i],
                        padding=padding[i],
                        bias=False
                    )
                )
                if batch_norm:
                    self.conv_layers.append(nn.BatchNorm2d(channels[i]))

        if self.dropout:
            self.conv_layers.append(nn.Dropout(0.25))
        self.conv_layers.append(self.get_activation_fn())

        ker_size = 4
        stride = ker_size
        pad = 0
        if pooling == "avg":
            self.conv_layers.append(
                nn.AvgPool2d(
                    kernel_size=ker_size,
                    padding=pad,
                    ceil_mode=False
                )
            )
            pooled_h = (shape_per_layer[-1][1] - ker_size) // stride + 1
            pooled_w = (shape_per_layer[-1][2] - ker_size) // stride + 1
            in_features = shape_per_layer[-1][0] * pooled_h * pooled_w
            self.fc_layers.append(
                nn.Linear(
                    in_features=in_features,
                    out_features=fc[0] if isinstance(fc, tuple) else fc,
                    bias=bias
                )
            )
        elif pooling == "max":
            self.conv_layers.append(
                nn.MaxPool2d(
                    kernel_size=ker_size,
                    padding=pad,
                    stride=stride,
                    return_indices=True
                )
            )
            pooled_h = (shape_per_layer[-1][1] - ker_size) // stride + 1
            pooled_w = (shape_per_layer[-1][2] - ker_size) // stride + 1
            in_features = shape_per_layer[-1][0] * pooled_h * pooled_w

            self.fc_layers.append(
                nn.Linear(
                    in_features=in_features,
                    out_features=fc[0] if isinstance(fc, tuple) else fc,
                    bias=bias
                )
            )

        if isinstance(fc, tuple):
            for i in range(1, len(fc)):
                self.fc_layers.append(self.get_activation_fn())
                if dropout:
                    self.fc_layers.append(nn.Dropout(0.5))
                self.fc_layers.append(
                    nn.Linear(
                        in_features=fc[i - 1],
                        out_features=fc[i],
                        bias=bias
                    )
                )
        self.fc_layers.append(self.get_activation_fn())
        self.fc_layers.append(
            nn.Linear(
                in_features=fc[-1] if isinstance(fc, tuple) else fc,
                out_features=num_classes,
                bias=bias
            )
        )

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def forward(self, x: torch.Tensor, rep: bool = False, return_penultimate: bool = False) -> torch.Tensor:
        """
        Forward pass for the CNN.

        Args:
            x (torch.Tensor): Input tensor
            rep (bool): Whether to save the activations and preactivations
            return_penultimate (bool): Whether to return features from the penultimate layer
        Returns:
            torch.Tensor: Output tensor (either final predictions or penultimate features).
        """
        if not rep:
            # Regular forward pass
            if self.batch_norm and (len(x.shape) == 3): x = x.unsqueeze(0)
            cnt = 0
            x_res = {}
            if cnt in self.residual: x_res[self.residual[cnt]] = x
            for layer in self.conv_layers:
                if isinstance(layer, nn.Conv2d):
                    cnt += 1
                elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.ELU, nn.LeakyReLU, nn.PReLU, nn.Sigmoid)):
                    if cnt in x_res:
                        x = x + x_res[cnt]
                        del x_res[cnt]
                    if cnt in self.residual:
                        if self.residual[cnt] not in x_res:
                            x_res[self.residual[cnt]] = x
                        else:
                            x_res[self.residual[cnt]] += x
                if isinstance(layer, nn.MaxPool2d):
                    x, _ = layer(x)
                else:
                    x = layer(x)
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
            else:
                x = torch.flatten(x)
            for layer in self.fc_layers[:-1]:
                if isinstance(layer, nn.Linear):
                    cnt += 1
                elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.ELU, nn.LeakyReLU, nn.PReLU, nn.Sigmoid)):
                    if cnt in x_res:
                        x = x + x_res[cnt]
                        del x_res[cnt]
                    if cnt in self.residual:
                        if self.residual[cnt] not in x_res:
                            x_res[self.residual[cnt]] = x
                        else:
                            x_res[self.residual[cnt]] += x
                if isinstance(layer, nn.MaxPool2d):
                    x, _ = layer(x)
                else:
                    x = layer(x)
            if return_penultimate:
                return x
            x = self.fc_layers[-1](x)
            return x

        # Forward pass for matrix computation
        # Save activations and preactivations
        self.pre_acts = []
        self.acts = []
        cnt = 0
        x_res = {}

        x = x.unsqueeze(0)
        if cnt in self.residual: x_res[self.residual[cnt]] = x
        x = self.conv_layers[0](x)  # Conv
        cnt += 1

        if self.save:
            self.pre_acts.append(x.detach().clone())
            if self.batch_norm:
                self.acts.append(x.detach().clone())
        if self.batch_norm:
            x = self.conv_layers[1](x)  # BN
            if self.save:
                self.pre_acts.append(x.detach().clone())
            if cnt in x_res:
                x = x + x_res[cnt]
                del x_res[cnt]
            if cnt in self.residual:
                if self.residual[cnt] not in x_res:
                    x_res[self.residual[cnt]] = x
                else:
                    x_res[self.residual[cnt]] += x
            x = self.conv_layers[2](x)  # Activation
            if self.save:
                self.acts.append(x.detach().clone())
        else:
            if cnt in x_res:
                x = x + x_res[cnt]
                del x_res[cnt]
            if cnt in self.residual:
                if self.residual[cnt] not in x_res:
                    x_res[self.residual[cnt]] = x
                else:
                    x_res[self.residual[cnt]] += x
            x = self.conv_layers[1](x)  # Activation
            if self.save:
                self.acts.append(x.detach().clone())

        for i in range(3 if self.batch_norm else 2, len(self.conv_layers)):
            layer = self.conv_layers[i]
            if isinstance(layer, nn.Conv2d):
                cnt += 1
                x = layer(x)
                if self.save:
                    self.pre_acts.append(x.detach().clone())
                    if self.batch_norm:
                        self.acts.append(x.detach().clone())

            elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.ELU, nn.LeakyReLU, nn.PReLU, nn.Sigmoid)):
                if cnt in x_res:
                    x = x + x_res[cnt]
                    del x_res[cnt]
                if cnt in self.residual:
                    if self.residual[cnt] not in x_res:
                        x_res[self.residual[cnt]] = x
                    else:
                        x_res[self.residual[cnt]] += x
                x = layer(x)
                if self.save:
                    self.acts.append(x.detach().clone())

            elif isinstance(layer, nn.AvgPool2d):
                x = layer(x)
                if self.save:
                    self.acts.append(x.detach().clone())
                    self.pre_acts.append(x.detach().clone())

            elif isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                if self.save:
                    self.acts.append(indices)
                    self.pre_acts.append(indices)

            elif isinstance(layer, nn.BatchNorm2d):
                x = layer(x)
                if self.save:
                    self.pre_acts.append(x.detach().clone())

        x = torch.flatten(x)

        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                cnt += 1
                x = layer(x)
                if self.save and layer != self.fc_layers[-1]:
                    self.pre_acts.append(x.detach().clone())
            elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.ELU, nn.LeakyReLU, nn.PReLU, nn.Sigmoid)):
                if cnt in x_res:
                    x = x + x_res[cnt]
                    del x_res[cnt]
                if cnt in self.residual:
                    if self.residual[cnt] not in x_res:
                        x_res[self.residual[cnt]] = x
                    else:
                        x_res[self.residual[cnt]] += x
                x = layer(x)
                if self.save:
                    self.acts.append(x.detach().clone())
        return x

        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.conv_layers = nn.ModuleList(self.conv_layers)

    def get_biases(self):
        """
        Get the biases of the linear layers.

        Returns:
            list[torch.Tensor]: List of biases
        """
        b = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                b.append(m.bias.data)
        return b

    def get_activation_fn(self):
        """
        Get the activation function.

        Returns:
            nn.Module: Activation function
        """
        act_name = self.activation.lower()
        activation_fn_map = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "leakyrelu": nn.LeakyReLU(),
            "prelu": nn.PReLU(),
            "sigmoid": nn.Sigmoid(),
        }
        if act_name not in activation_fn_map.keys():
            raise ValueError(f"Unknown activation function name : {act_name}")
        return activation_fn_map[act_name]

    def init(self) -> None:
        """
        Initialize the weights of the model.
        """

        def init_func(m) -> None:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.running_mean = torch.randn(m.num_features)
                m.running_var = torch.rand(m.num_features)
            #    m.reset_parameters()

        self.apply(init_func)

    @staticmethod
    def round_up(n: float) -> int:
        """
        Round up a number to the nearest integer.

        Args:
            n (float): Number to round up
        Returns:
            int: Nearest integer
        """
        # TODO: There are inconsistencies in the output shape of AvgPool2d.
        # Will need to look into it in the future
        return math.floor(n)
        if n - math.floor(n) < 0.5:
            return math.floor(n)
        return math.ceil(n)