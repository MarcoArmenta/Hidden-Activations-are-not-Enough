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
from model_zoo.res_net import ResNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)

def get_memory_usage() -> float:
    """
        Returns:
            Current memory usage in MB
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class TestResNetRepresentation(unittest.TestCase):
    def generate_random_params(self) -> tuple[int, int]:
        """
            Returns:
                w: width of the input image
                num_classes: number of classes in the output layer
        """
        w = random.randint(20,35)
        num_classes = random.randint(10,20)
        return w, num_classes

    def create_random_model(self) -> tuple[ResNet, torch.Tensor, torch.Tensor, int] :
        """
            Returns:
                model: the ResNet model
                x: the input tensor
                forward_pass: the output of the forward pass
                num_classes: number of classes in the output layer
        """
        w, num_classes = self.generate_random_params()
        input_shape = (random.randint(1,3),w,w)
        x = torch.rand(input_shape)

        model = ResNet(
            input_shape = input_shape,
            num_classes = num_classes,
            pretrained = True,
            # TODO: Currently doesn't work if max_pool=True and pretrained=True. 
            # Should work otherwise.
            max_pool = False
        ).to(DEVICE)
        model.eval()
        model.save = True
        forward_pass = model(x)

        return model, x, forward_pass, num_classes

    def test_ResNetRepBuild(self) -> None:
        """
            Test if building the matrix keeps the network function unchanged.
            Test memory usage and time taken.
        """
        for test_num in range(5):
            print(f"\n--------------------------------")
            print(f"Test {test_num + 1}/5:")

            # Clear memory before each test
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start_time = time()
            start_mem = get_memory_usage()

            model, x, forward_pass, num_classes = self.create_random_model()
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
            print(f"  -  Batch Size: {batch_size}")
            print(f"Performance:")
            print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
            print(f"--------------------------------")


if __name__ == "__main__":
    unittest.main()
