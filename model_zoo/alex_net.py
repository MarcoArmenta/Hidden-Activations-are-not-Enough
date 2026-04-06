import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights


class AlexNet(nn.Module):
    """
    Implementation of AlexNet architecture with configurable input shape and number of classes.
    Can be initialized with pretrained weights and supports saving activations for matrix computation.

    Args:
        input_shape (tuple[int,int,int]): Input shape as (channels, height, width)
        num_classes (int): Number of output classes
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        max_pool (bool, optional): Whether to use max pooling layers. Defaults to True.
        save (bool, optional): Whether to save activations during forward pass. Defaults to False.
    """
    def __init__(
            self,
            input_shape,
            num_classes: int,
            pretrained:bool = True,
            freeze_features:bool = True,
            max_pool:bool = True,
            save:bool = False
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        c,h,w = input_shape
        self.num_classes = num_classes
        self.bias = True
        self.save = save

        if pretrained:
            self.input_shape = (3, 224, 224)
            self.num_classes = 1000
            self.weights = AlexNet_Weights.DEFAULT
        else:
            self.weights = None

        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = alexnet(weights=self.weights, progress=False)
        #self.model.eval()
        #self.model.to(self.device)
        self.matrix_input_dim = c*w*h + 1
        if num_classes != 1000:
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes, bias=True)
        regular_input = h >= 224
        if regular_input:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406], 
                    std = [0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406], 
                    std = [0.229, 0.224, 0.225]
                )
            ])

        self.conv_layers = []
        self.fc_layers = []
        
        # Iterate through model layers and build conv_layers and fc_layers lists
        # Handles MaxPool2d, Linear, Dropout, Conv2d and other layer types
        # Adapts first conv layer and final linear layer based on input shape and num_classes
        first_conv = True
        fc = 0
        for module in self.model.children():
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.MaxPool2d):
                        if max_pool:
                            if regular_input:
                                self.conv_layers.append(
                                    nn.MaxPool2d(
                                        kernel_size = layer.kernel_size,
                                        padding = layer.padding,
                                        stride = layer.stride,
                                        return_indices = True
                                    )
                                )
                            else:
                                self.conv_layers.append(
                                    nn.MaxPool2d(
                                        kernel_size = 2,
                                        padding = 0,
                                        stride = 1,
                                        return_indices = True
                                    )
                                )

                    elif isinstance(layer, nn.Linear):
                        if num_classes != 1000:
                            if layer.out_features != num_classes:
                                self.fc_layers.append(
                                    nn.Linear(
                                        in_features = layer.in_features,
                                        out_features = layer.out_features,
                                        bias = True
                                    )
                                )
                            else:
                                self.fc_layers.append(
                                    nn.Linear(
                                        in_features=layer.in_features,
                                        out_features=num_classes,
                                        bias=True
                                    )
                                )
                        else:
                            self.fc_layers.append(layer)
                        fc += 1
                    
                    elif isinstance(layer, nn.Dropout):
                        self.fc_layers.append(layer)
                        fc += 1

                    elif isinstance(layer, nn.Conv2d):
                        if first_conv:
                            if c == 3 and regular_input:
                                self.conv_layers.append(layer)
                            else:
                                self.conv_layers.append(
                                    nn.Conv2d(
                                        in_channels = input_shape[0], 
                                        out_channels = 64, 
                                        kernel_size = 6, 
                                        stride = 3, 
                                        padding = 1, 
                                        bias = False
                                    )
                                )
                            first_conv = False
                        else:
                            self.conv_layers.append(layer)

                    else:
                        if fc == 0:  # Not yet in the fully connected part of the NN
                            self.conv_layers.append(layer)
                        else:
                            self.fc_layers.append(layer)
            
            elif isinstance(module, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                self.conv_layers.append(module)

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.fc_layers = nn.ModuleList(self.fc_layers)

        if pretrained and freeze_features:
            for layer in self.conv_layers:
                if isinstance(layer, nn.Conv2d):
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(self, x: torch.Tensor, rep:bool=False, return_penultimate:bool=False) -> torch.Tensor:
        """
        Forward pass of the AlexNet model.

        Args:
            x (torch.Tensor): Input tensor
            rep (bool, optional): Whether to save activations. Defaults to False.
            return_penultimate (bool, optional): Whether to return features from the penultimate layer.
        Returns:
            torch.Tensor: Output tensor (either final predictions or penultimate features).
        """
        if not rep:
            # Regular forward pass
            if len(x.shape) == 3: x = x.unsqueeze(0)

            for layer in self.conv_layers:
                if isinstance(layer, nn.MaxPool2d):
                    x, _ = layer(x)
                else:
                    x = layer(x)
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
            else:
                x = torch.flatten(x)

            for layer in self.fc_layers[:-1]:
                x = layer(x)

            if return_penultimate: 
                return x
            x = self.fc_layers[-1](x)
            return x

        # Forward pass for matrix computation
        # Save activations and preactivations
        self.pre_acts = []
        self.acts = []

        x = x.unsqueeze(0)
        x = self.conv_layers[0](x)  # Conv
        if self.save:
            self.pre_acts.append(x.detach().clone())
        x = self.conv_layers[1](x)  # Activation
        if self.save:
            self.acts.append(x.detach().clone())

        for i in range(2, len(self.conv_layers)):
            layer = self.conv_layers[i]

            if isinstance(layer, nn.Conv2d):
                x = layer(x)
                if self.save:
                    self.pre_acts.append(x.detach().clone())
            elif isinstance(layer, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                x = layer(x)
                if self.save:
                    self.pre_acts.append(x.detach().clone())
                    self.acts.append(x.detach().clone())
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
                if self.save:
                    self.acts.append(x.detach().clone())
            elif isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                if self.save:
                    self.acts.append(indices)
                    self.pre_acts.append(indices)

        x = torch.flatten(x)

        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if self.save and layer != self.fc_layers[-1]:
                    self.pre_acts.append(x.detach().clone())
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
                if self.save:
                    self.acts.append(x.detach().clone())
        return x
    
    def get_activation_fn(self):
        """
        Returns the activation function used in the model.

        Returns:
            nn.ReLU: ReLU activation function
        """
        return nn.ReLU()
