#!/usr/bin/env python
"""
    Test if building the representation keeps the network function unchanged
"""
import sys
import os
import unittest
import torch
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matrix_construction.representation import MlpRepresentation
from model_zoo.mlp import MLP

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


class TestMLPRepresentation(unittest.TestCase):

    def generate_random_params(self):
        w = random.randint(28, 32)
        l = random.randint(1, 200)
        c = random.randint(1, 1000)
        num_classes = random.randint(1, 10)
        return w, l, c, num_classes

    def create_random_model(self):
        w, l, c, num_classes = self.generate_random_params()
        input_shape = (3, w, w)
        x = torch.rand(input_shape)

        model = MLP(input_shape=input_shape,
                    num_classes=num_classes,
                    hidden_sizes=tuple(c for _ in range(l)),
                    bias=True,
                    residual=False,
                    ).to(DEVICE)
        model.init()
        model.eval()
        model.save = True
        forward_pass = model(x)

        return model, x, forward_pass, w, l, c, num_classes

    def test_MLPRepBuild(self):
        for _ in range(1000):
            model, x, forward_pass, w, l, c, num_classes = self.create_random_model()

            # Build representation and compute output
            rep = MlpRepresentation(model, DEVICE)
            rep = rep.forward(x)
            one = torch.flatten(torch.ones(model.matrix_input_dim))
            rep_forward = torch.matmul(rep, one)
            diff = torch.norm(rep_forward - forward_pass).detach().numpy()

            self.assertAlmostEqual(diff, 0, places=None, msg=f"rep and forward_pass differ by {diff}.", delta=0.1)
            print(f"Test passed for w={w}, l={l}, c={c}, num_classes={num_classes}")


if __name__ == "__main__":
    unittest.main()
