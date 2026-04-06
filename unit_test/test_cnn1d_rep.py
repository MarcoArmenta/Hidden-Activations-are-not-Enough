#!/usr/bin/env python
import sys
import os
import unittest
import torch
import random
import psutil
import gc
from time import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matrix_construction.matrix_computation import ConvRepresentation_2D
from model_zoo.cnn import CNN_2D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)

def get_memory_usage() -> float:
    """
        Returns:
            Current memory usage in MB
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class TestConvRepresentation_2D(unittest.TestCase):
    def generate_random_params(self) -> tuple[int, tuple[int], int, int]:
        """
            Returns:
                w: width of the input image
                channels: tuple of the number of channels in each convolutional layer
                fc: number of neurons in the (only) fully connected layer
                num_classes: number of classes in the output layer
        """
        w = random.randint(20,60)
        channels = tuple(random.randint(3,50) for _ in range(random.randint(3,8)))
        fc = random.randint(200,400)
        num_classes = random.randint(5,15)
        return w, channels, fc, num_classes

    def create_random_model(self) -> tuple[CNN_2D, torch.Tensor, torch.Tensor, tuple[int], int, int]:
        """
            Returns:
                model: the CNN model
                x: the input tensor
                forward_pass: the output of the forward pass
                channels: tuple of the number of channels in each convolutional layer
                fc: number of neurons in the (only) fully connected layer
                num_classes: number of classes in the output layer
        """
        w, channels, fc, num_classes = self.generate_random_params()
        padding = tuple((0,1) for _ in channels)  # Should be fixed for now.
        kernel = tuple((1,3) for _ in channels)
        input_shape = (random.randint(1,4),1,w)
        x = torch.rand(input_shape)

        model = CNN_2D(
            input_shape = input_shape,
            num_classes = num_classes,
            channels = channels,
            padding = padding,
            kernel_size = kernel,
            kernel_pooling = (1,4),
            fc = fc,
            bias = False,
            # List of pairs (start_index, end_index), of the residual connections.
            residual = [],
            batch_norm = False,
            activation = "relu",
            pooling = "avg"
        ).to(DEVICE)
        model.eval()
        model.save = True
        forward_pass = model(x)

        return model, x, forward_pass, channels, fc, num_classes

    def test_ConvRepBuild(self) -> None:
        """
            Test if building the matrix keeps the network function unchanged.
            Test memory usage and time taken.
        """
        for test_num in range(10):
            print(f"\n--------------------------------")
            print(f"Test {test_num + 1}/10:")

            # Clear memory before each test
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start_mem = get_memory_usage()
            start_time = time()

            model, x, forward_pass, channels, fc, num_classes = self.create_random_model()
            batch_size = random.randint(1,16)

            # Build representation and compute output
            rep = ConvRepresentation_2D(model, batch_size=batch_size)
            rep = rep.forward(x)

            one = torch.flatten(torch.ones(model.matrix_input_dim))
            rep_forward = torch.matmul(rep, one)
            diff = torch.norm(rep_forward - forward_pass).item()

            end_time = time()
            end_mem = get_memory_usage()
            mem_used = end_mem - start_mem

            self.assertAlmostEqual(
                first = diff, 
                second = 0, 
                places = None, 
                msg = f"rep and forward_pass differ by {diff}.", 
                delta = 0.1
            )

            print(f"Results:")
            print(f"Model Architecture:")
            print(f"  - Input Shape: ({x.shape[0]}, {x.shape[1]}, {x.shape[2]})  - Output Classes: {num_classes}")
            print(f"  - Channels: {channels}  -  FC Layer: {fc}")
            print(f"  -  Batch Size: {batch_size}")
            print(f"Performance:")
            print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
            print(f"--------------------------------")


if __name__ == "__main__":
    unittest.main()
