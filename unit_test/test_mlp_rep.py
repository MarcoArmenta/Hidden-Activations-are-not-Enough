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

from matrix_construction.matrix_computation import MlpRepresentation
from model_zoo.mlp import MLP

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)

def get_memory_usage() -> float:
    """
        Returns:
            Current memory usage in MB
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class TestMLPRepresentation(unittest.TestCase):
    def generate_random_params(self) -> tuple[int, int, int, int]:
        """
            Returns:
                w: width of the input image
                l: number of layers
                c: number of neurons per layer
                num_classes: number of classes in the output layer
        """
        w = random.randint(28, 32)
        l = random.randint(1, 50)
        c = random.randint(1, 800)
        num_classes = random.randint(2, 100)
        return w, l, c, num_classes

    def create_random_model(self) -> tuple[MLP, torch.Tensor, torch.Tensor, int, int, int]:
        """
            Returns:
                model: the MLP model
                x: the input tensor
                forward_pass: the output of the forward pass
                l: number of layers
                c: number of neurons per layer
                num_classes: number of classes in the output layer
        """
        w, l, c, num_classes = self.generate_random_params()
        input_shape = (3, w, w)
        x = torch.rand(input_shape)

        model = MLP(
            input_shape = input_shape,
            num_classes = num_classes,
            hidden_sizes = tuple(c for _ in range(l)),
            bias = True,
            residual = False
        ).to(DEVICE)
        model.eval()
        model.save = True
        forward_pass = model(x)

        return model, x, forward_pass, l, c, num_classes

    def test_MLPRepBuild(self) -> None:
        """
            Test if building the matrix keeps the network function unchanged.
            Test memory usage and time taken.
        """
        for test_num in range(50):
            print(f"\n--------------------------------")
            print(f"Test {test_num + 1}/50:")

            # Clear memory before each test
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start_mem = get_memory_usage()
            start_time = time()

            model, x, forward_pass, l, c, num_classes = self.create_random_model()

            # Build representation and compute output
            rep = MlpRepresentation(model, build_rep=True, device=DEVICE)
            matrix = rep.forward(x)
            one = torch.flatten(torch.ones(model.matrix_input_dim))
            rep_forward = torch.matmul(matrix, one)
            diff = torch.norm(rep_forward - forward_pass).detach().numpy()

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
            print(f"  - Input Shape: ({x.shape[0]}, {x.shape[1]}, {x.shape[2]})")
            print(f"  - Number of Layers: {l}  -  Neurons per Layer: {c}  -  Output Classes: {num_classes}")
            print(f"Performance:")
            print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
            print(f"--------------------------------")


if __name__ == "__main__":
    unittest.main()
