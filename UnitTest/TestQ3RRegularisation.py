import unittest

import numpy as np
import matplotlib.pyplot as plt

import torch

from torch import tensor, nn

from UnitTest.Old_QuaRS import QuaRS
from UnitTest.Old_QuaRS import W_x as WxOld

from Functions.Q3R import QuaRS as QuaRS_v2
from Functions.Q3R import W_x_v2 as WxNew

from Functions.quars_v3 import wx as WxNewV2


class Q3R_unittest(unittest.TestCase):
    def initialization(self):
        try:
            layer1 = nn.Linear(300, 300)
            regularisation = QuaRS(trainable_weights=[layer1])
        except Exception as e:
            self.fail(f"Initialization failed with error: {e}")

    def gpu_intialization(self):
        DEVICE = torch.device("cuda")
        try:
            layer1 = nn.Linear(300, 300)
            layer1.to(device=DEVICE)
            regularisation = QuaRS(trainable_weights=[layer1])
        except Exception as e:
            self.fail(f"Initialization failed with error: {e}")


class RegularisationConversionUnitTest(unittest.TestCase):
    class BasicModel(nn.Module):
        def __init__(self, input_size, output_size, layers=1):
            super().__init__()

            if layers == 1:
                self.model = nn.Sequential(*[nn.Linear(input_size, output_size)])
            elif layers == 2:
                self.model = nn.Sequential(*[nn.Linear(input_size, 100), nn.Linear(100, output_size)])
            else:
                if layers <= 2:
                    raise ValueError("Layers cannot be less than 1")

                model_list = [nn.Linear(input_size, 100)] + [nn.Linear(100, 100) for i in range(layers - 2)] + [
                    nn.Linear(100, output_size)]
                self.model = nn.Sequential(*model_list)

        def forward(self, x):
            x = self.model(x)
            return x

    class BasicRectangularModel(nn.Module):
        def __init__(self, input_size, output_size=200, layers=1):
            super().__init__()

            if layers == 1:
                self.model = nn.Sequential(*[nn.Linear(input_size, output_size)])
            elif layers == 2:
                self.model = nn.Sequential(*[nn.Linear(input_size, 200), nn.Linear(200, output_size)])
            else:
                if layers <= 2:
                    raise ValueError("Layers cannot be less than 1")

                model_list = [nn.Linear(input_size, 200)] + [nn.Linear(200, 100) if i % 2 == 0 else nn.Linear(100, 200)
                                                             for i in range(layers - 2)] + [
                                 nn.Linear(200, output_size) if layers % 2 == 0 else nn.Linear(100, output_size)]
                self.model = nn.Sequential(*model_list)

        def forward(self, x):
            x = self.model(x)
            return x

    class BasicModelOld(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, output_size)

        def forward(self, x):
            x = self.fc1(x)
            return x

    def test_square_convergence_single_layer(self):
        LR = 0.0001
        lmbda = 1
        dim = (100, 100)
        target_rank = 10
        layers = 1
        for j in range(1):
            torch.manual_seed(j)

            model_A, model_B = self.BasicModel(dim[0], dim[1], layers=1), self.BasicModel(dim[0], dim[1], layers=1)
            print(type(model_A))
            print(model_A)
            print(model_A.parameters())

            for index in range(layers):
                init_matrix = torch.rand(dim)
                model_A.model[index].weight.data = init_matrix.clone()
                model_B.model[index].weight.data = init_matrix.clone()
            print(model_A, model_B)

            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Pre Model A Layer:{index}")

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Pre Model B Layer:{index}")

            old_q3r = QuaRS(trainable_weights=[layer.weight for layer in model_A.model], target_rank=target_rank,
                            lmbda=lmbda, steps=1,
                            rectangular_mode=False, verbose=True)
            new_q3r = QuaRS_v2(trainable_weights=[layer for layer in model_B.model], target_rank=target_rank,
                               lmbda=lmbda, steps=1,
                               rectangular_mode=False, verbose=True, scaling=1)

            optimizer_A, optimizer_B = torch.optim.SGD(model_A.parameters(), lr=LR), torch.optim.SGD(
                model_B.parameters(), lr=LR)

            for i in range(1000):
                optimizer_A.zero_grad(), optimizer_B.zero_grad()
                print("new"), new_q3r.update(), print("old"), old_q3r.update()

                new_q3r.val.backward(retain_graph=True), old_q3r.val.backward(retain_graph=True)

                optimizer_A.step(), optimizer_B.step()

                """self.assertAlmostEqual(old_q3r.val.item(), new_q3r.val.item(),
                                       places=-3, msg=f"Values diverged at iteration {i}")"""

            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Model A Layer:{index}")

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Model B Layer:{index}")

            plt.legend()
            plt.grid(True)
            plt.show()

    def test_old_square_convergence_single_layer(self):
        LR = 0.01
        lmbda = 1
        dim = (100, 100)
        target_rank = 10

        for j in range(1):
            torch.manual_seed(j)

            init_matrix = torch.rand(dim)

            model_A, model_B = self.BasicModelOld(dim[0], dim[1]), self.BasicModelOld(dim[0], dim[1])

            model_A.fc1.weight.data = init_matrix.clone()
            model_B.fc1.weight.data = init_matrix.clone()

            old_q3r = QuaRS(trainable_weights=[model_A.fc1.weight], target_rank=target_rank, lmbda=lmbda,
                            steps=5,
                            rectangular_mode=False, verbose=True)
            new_q3r = QuaRS_v2(trainable_weights=[model_B.fc1], target_rank=target_rank, lmbda=lmbda,
                               steps=5,
                               rectangular_mode=False, verbose=True, scaling=1)

            optimizer_A, optimizer_B = torch.optim.SGD(model_A.parameters(), lr=LR), torch.optim.SGD(
                model_B.parameters(), lr=LR)

            for i in range(10000):
                optimizer_A.zero_grad(), optimizer_B.zero_grad()
                print("new"), new_q3r.update(), print("old"), old_q3r.update()

                new_q3r.val.backward(), old_q3r.val.backward(retain_graph=True)

                optimizer_A.step(), optimizer_B.step()

                self.assertAlmostEqual(old_q3r.val.item(), new_q3r.val.item(),
                                       places=-3, msg=f"Values diverged at iteration {i}")

            A_s = torch.svd(model_A.fc1.weight.data)[1]
            plt.plot(np.arange(len(A_s)), A_s, label=f"Model A, Layer 1")

            B_s = torch.svd(model_B.fc1.weight.data)[1]
            plt.plot(np.arange(len(B_s)), B_s, label=f"Model B, Layer 1")

            plt.legend()
            plt.grid(True)
            plt.show()

    def test_square_convergence_two_layer(self):
        LR = 0.001
        lmbda = 1
        dim = (100, 100)
        target_rank = 10
        layers = 2
        for j in range(1):
            torch.manual_seed(j)

            model_A, model_B = self.BasicModel(dim[0], dim[1], layers=layers), self.BasicModel(dim[0], dim[1],
                                                                                               layers=layers)

            for index in range(layers):
                init_matrix = torch.rand(dim)
                model_A.model[index].weight.data = init_matrix.clone()
                model_B.model[index].weight.data = init_matrix.clone()
            print(model_A, model_B)

            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Pre Model A Layer:{index}")

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Pre Model B Layer:{index}")

            old_q3r = QuaRS(trainable_weights=[layer.weight for layer in model_A.model], target_rank=target_rank,
                            lmbda=lmbda, steps=5,
                            rectangular_mode=False, verbose=True)
            new_q3r = QuaRS_v2(trainable_weights=[layer for layer in model_B.model], target_rank=target_rank,
                               lmbda=lmbda, steps=5,
                               rectangular_mode=False, verbose=True, scaling=1)

            optimizer_A, optimizer_B = torch.optim.SGD(model_A.parameters(), lr=LR), torch.optim.SGD(
                model_B.parameters(), lr=LR)

            for i in range(100):
                optimizer_A.zero_grad(), optimizer_B.zero_grad()
                print("new"), new_q3r.update(), print("old"), old_q3r.update()

                new_q3r.val.backward(retain_graph=True), old_q3r.val.backward(retain_graph=True)

                optimizer_A.step(), optimizer_B.step()

                """self.assertAlmostEqual(old_q3r.val.item(), new_q3r.val.item(),
                                       places=-3, msg=f"Values diverged at iteration {i}")"""

            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Model A Layer:{index}", color='blue')

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Model B Layer:{index}", color='red')

            plt.legend()
            plt.grid(True)
            plt.show()

    def test_square_convergence_ten_layer(self):
        LR = 0.01
        lmbda = 1
        dim = (100, 100)
        target_rank = 10
        layers = 10
        for j in range(1):
            torch.manual_seed(j)

            model_A, model_B = self.BasicModel(dim[0], dim[1], layers=layers), self.BasicModel(dim[0], dim[1],
                                                                                               layers=layers)

            for index in range(layers):
                init_matrix = torch.rand(dim)
                model_A.model[index].weight.data = init_matrix.clone()
                model_B.model[index].weight.data = init_matrix.clone()
            print(model_A, model_B)

            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Pre Model A Layer:{index}")

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Pre Model B Layer:{index}")

            old_q3r = QuaRS(trainable_weights=[layer.weight for layer in model_A.model], target_rank=target_rank,
                            lmbda=lmbda, steps=5,
                            rectangular_mode=False, verbose=True)
            new_q3r = QuaRS_v2(trainable_weights=[layer for layer in model_B.model], target_rank=target_rank,
                               lmbda=lmbda, steps=5,
                               rectangular_mode=False, verbose=True, scaling=1)

            optimizer_A, optimizer_B = torch.optim.SGD(model_A.parameters(), lr=LR), torch.optim.SGD(
                model_B.parameters(), lr=LR)

            for i in range(100):
                optimizer_A.zero_grad(), optimizer_B.zero_grad()
                print("new"), new_q3r.update(), print("old"), old_q3r.update()

                new_q3r.val.backward(retain_graph=True), old_q3r.val.backward(retain_graph=True)

                optimizer_A.step(), optimizer_B.step()

                """self.assertAlmostEqual(old_q3r.val.item(), new_q3r.val.item(),
                                       places=-3, msg=f"Values diverged at iteration {i}")"""

            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Model A Layer:{index}", color='blue')

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Model B Layer:{index}", color='red')

            plt.legend()
            plt.grid(True)
            plt.show()

    def test_rectangle_convergence_single_layer(self):
        # Rectangle Matrix with Square Mode
        LR = 0.01
        lmbda = 1
        dim = (100, 200)
        target_rank = 20
        layers = 1
        for j in range(1):
            torch.manual_seed(j)

            model_A, model_B = self.BasicRectangularModel(dim[0], dim[1], layers=1), self.BasicRectangularModel(dim[0],
                                                                                                                dim[1],
                                                                                                                layers=1)

            for index in range(layers):
                init_matrix = torch.rand(dim)
                model_A.model[index].weight.data = init_matrix.clone()
                model_B.model[index].weight.data = init_matrix.clone()
            print(model_A, model_B)

            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Pre Model A Layer:{index}")

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Pre Model B Layer:{index}")

            old_q3r = QuaRS(trainable_weights=[layer.weight for layer in model_A.model], target_rank=target_rank,
                            lmbda=lmbda, steps=5,
                            rectangular_mode=False, verbose=True)
            new_q3r = QuaRS_v2(trainable_weights=[layer for layer in model_B.model], target_rank=target_rank,
                               lmbda=lmbda, steps=5,
                               rectangular_mode=False, verbose=True, scaling=1)

            optimizer_A, optimizer_B = torch.optim.SGD(model_A.parameters(), lr=LR), torch.optim.SGD(
                model_B.parameters(), lr=LR)

            for i in range(100):
                optimizer_A.zero_grad(), optimizer_B.zero_grad()
                print("new"), new_q3r.update(), print("old"), old_q3r.update()

                new_q3r.val.backward(retain_graph=True), old_q3r.val.backward(retain_graph=True)

                optimizer_A.step(), optimizer_B.step()

                """self.assertAlmostEqual(old_q3r.val.item(), new_q3r.val.item(),
                                       places=-3, msg=f"Values diverged at iteration {i}")
                self.assertEquals([wx.rectangular_mode for wx in old_q3r.regularizers], [False])
                self.assertEquals([wx.rectangular_mode for wx in new_q3r.regularizers], [False])"""
            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Model A Layer:{index}", color='blue')

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Model B Layer:{index}", color='red')

            plt.legend()
            plt.grid(True)
            plt.show()

    def test_rectangle_mode2_convergence_single_layer(self):
        # Rectangle Matrix with Rectangular Mode 2
        LR = 0.01
        lmbda = 1
        dim = (100, 200)
        target_rank = 20
        layers = 1
        for j in range(1):
            torch.manual_seed(j)

            model_A, model_B = self.BasicRectangularModel(dim[0], dim[1], layers=1), self.BasicRectangularModel(dim[0],
                                                                                                                dim[1],
                                                                                                                layers=1)

            for index in range(layers):
                init_matrix = torch.rand(dim)
                model_A.model[index].weight.data = init_matrix.clone()
                model_B.model[index].weight.data = init_matrix.clone()
            print(model_A, model_B)

            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Pre Model A Layer:{index}")

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Pre Model B Layer:{index}")

            old_q3r = QuaRS(trainable_weights=[layer.weight for layer in model_A.model], target_rank=target_rank,
                            lmbda=lmbda, steps=5,
                            rectangular_mode=2, verbose=True)
            new_q3r = QuaRS_v2(trainable_weights=[layer for layer in model_B.model], target_rank=target_rank,
                               lmbda=lmbda, steps=5,
                               rectangular_mode=2, verbose=True, scaling=1)

            optimizer_A, optimizer_B = torch.optim.SGD(model_A.parameters(), lr=LR), torch.optim.SGD(
                model_B.parameters(), lr=LR)

            for i in range(100):
                optimizer_A.zero_grad(), optimizer_B.zero_grad()
                print("new"), new_q3r.update(), print("old"), old_q3r.update()

                new_q3r.val.backward(retain_graph=True), old_q3r.val.backward(retain_graph=True)

                optimizer_A.step(), optimizer_B.step()

                """self.assertAlmostEqual(old_q3r.val.item(), new_q3r.val.item(),
                                       places=-3, msg=f"Values diverged at iteration {i}")
                self.assertEquals([wx.rectangular_mode for wx in old_q3r.regularizers], [2])
                self.assertEquals([wx.rectangular_mode for wx in new_q3r.regularizers], [2])"""
            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Model A Layer:{index}", color='blue')

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Model B Layer:{index}", color='red')
            plt.title("test_rectangle_mode2_convergence_single_layer")
            plt.legend()
            plt.grid(True)
            plt.show()

    def test_rectangle_convergence_two_layer(self):
        # Rectangle Matrix with Square Mode
        LR = 0.01
        lmbda = 1
        dim = (100, 200)
        target_rank = 20
        layers = 2
        for j in range(1):
            torch.manual_seed(j)

            model_A, model_B = self.BasicRectangularModel(dim[0], dim[1], layers=layers), self.BasicRectangularModel(
                dim[0],
                dim[1],
                layers=layers)

            for index in range(layers):
                init_matrix = torch.rand(dim)
                model_A.model[index].weight.data = init_matrix.clone()
                model_B.model[index].weight.data = init_matrix.clone()
            print(model_A, model_B)

            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Pre Model A Layer:{index}")

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Pre Model B Layer:{index}")

            old_q3r = QuaRS(trainable_weights=[layer.weight for layer in model_A.model], target_rank=target_rank,
                            lmbda=lmbda, steps=5,
                            rectangular_mode=False, verbose=True)
            new_q3r = QuaRS_v2(trainable_weights=[layer for layer in model_B.model], target_rank=target_rank,
                               lmbda=lmbda, steps=5,
                               rectangular_mode=False, verbose=True, scaling=1)

            optimizer_A, optimizer_B = torch.optim.SGD(model_A.parameters(), lr=LR), torch.optim.SGD(
                model_B.parameters(), lr=LR)

            for i in range(100):
                optimizer_A.zero_grad(), optimizer_B.zero_grad()
                print("new"), new_q3r.update(), print("old"), old_q3r.update()

                new_q3r.val.backward(retain_graph=True), old_q3r.val.backward(retain_graph=True)

                optimizer_A.step(), optimizer_B.step()

                self.assertAlmostEqual(old_q3r.val.item(), new_q3r.val.item(),
                                       places=-3, msg=f"Values diverged at iteration {i}")
                self.assertEquals([wx.rectangular_mode for wx in old_q3r.regularizers], [False, False])
                self.assertEquals([wx.rectangular_mode for wx in new_q3r.regularizers], [False, False])
            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Model A Layer:{index}", color='blue')

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Model B Layer:{index}", color='red')
            plt.title('test_rectangle_convergence_two_layer')
            plt.legend()
            plt.grid(True)
            plt.show()

    def test_rectangle_convergence_two_layer_2000_steps(self):
        # Rectangle Matrix with Square Mode
        LR = 0.01
        lmbda = 1
        dim = (100, 200)
        target_rank = 20
        layers = 2
        for j in range(1):
            torch.manual_seed(j)

            model_A, model_B = self.BasicRectangularModel(dim[0], dim[1], layers=layers), self.BasicRectangularModel(
                dim[0],
                dim[1],
                layers=layers)

            for index in range(layers):
                init_matrix = torch.rand(dim)
                model_A.model[index].weight.data = init_matrix.clone()
                model_B.model[index].weight.data = init_matrix.clone()
            print(model_A, model_B)

            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Pre Model A Layer:{index}")

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Pre Model B Layer:{index}")

            old_q3r = QuaRS(trainable_weights=[layer.weight for layer in model_A.model], target_rank=target_rank,
                            lmbda=lmbda, steps=5,
                            rectangular_mode=False, verbose=True)
            new_q3r = QuaRS_v2(trainable_weights=[layer for layer in model_B.model], target_rank=target_rank,
                               lmbda=lmbda, steps=5,
                               rectangular_mode=False, verbose=True, scaling=1)

            optimizer_A, optimizer_B = torch.optim.SGD(model_A.parameters(), lr=LR), torch.optim.SGD(
                model_B.parameters(), lr=LR)

            for i in range(2000):
                optimizer_A.zero_grad(), optimizer_B.zero_grad()
                print("new"), new_q3r.update(), print("old"), old_q3r.update()

                new_q3r.val.backward(retain_graph=True), old_q3r.val.backward(retain_graph=True)

                optimizer_A.step(), optimizer_B.step()

                self.assertAlmostEqual(old_q3r.val.item(), new_q3r.val.item(),
                                       places=-3, msg=f"Values diverged at iteration {i}")
                self.assertEquals([wx.rectangular_mode for wx in old_q3r.regularizers], [False, False])
                self.assertEquals([wx.rectangular_mode for wx in new_q3r.regularizers], [False, False])
            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Model A Layer:{index}", color='blue')

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Model B Layer:{index}", color='red')
            plt.title('test_rectangle_convergence_two_layer_2000_steps')
            plt.legend()
            plt.grid(True)
            plt.show()

    def test_rectangle_mode2_convergence_two_layer(self):
        # Rectangle Matrix with Rectangular Mode 2
        LR = 0.01
        lmbda = 1
        dim = (100, 200)
        target_rank = 20
        layers = 2
        for j in range(1):
            torch.manual_seed(j)

            model_A, model_B = self.BasicRectangularModel(dim[0], dim[1], layers=layers), self.BasicRectangularModel(
                dim[0],
                dim[1],
                layers=layers)

            for index in range(layers):
                init_matrix = torch.rand(dim)
                model_A.model[index].weight.data = init_matrix.clone()
                model_B.model[index].weight.data = init_matrix.clone()
            print(model_A, model_B)

            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Pre Model A Layer:{index}")

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Pre Model B Layer:{index}")

            old_q3r = QuaRS(trainable_weights=[layer.weight for layer in model_A.model], target_rank=target_rank,
                            lmbda=lmbda, steps=5,
                            rectangular_mode=2, verbose=True)
            new_q3r = QuaRS_v2(trainable_weights=[layer for layer in model_B.model], target_rank=target_rank,
                               lmbda=lmbda, steps=5,
                               rectangular_mode=2, verbose=True, scaling=1)

            optimizer_A, optimizer_B = torch.optim.SGD(model_A.parameters(), lr=LR), torch.optim.SGD(
                model_B.parameters(), lr=LR)

            for i in range(100):
                optimizer_A.zero_grad(), optimizer_B.zero_grad()
                print("new"), new_q3r.update(), print("old"), old_q3r.update()

                new_q3r.val.backward(retain_graph=True), old_q3r.val.backward(retain_graph=True)

                optimizer_A.step(), optimizer_B.step()

                """self.assertAlmostEqual(old_q3r.val.item(), new_q3r.val.item(),
                                       places=-3, msg=f"Values diverged at iteration {i}")
                self.assertEquals([wx.rectangular_mode for wx in old_q3r.regularizers], [2,2])
                self.assertEquals([wx.rectangular_mode for wx in new_q3r.regularizers], [2,2])"""
            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Model A Layer:{index}", color='blue')

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Model B Layer:{index}", color='red')
            plt.title('test_rectangle_mode2_convergence_two_layer')
            plt.legend()
            plt.grid(True)
            plt.show()

    def test_rectangle_mode2_rank_ratio_intialization_two_layer(self):
        # Rectangle Matrix with Rectangular Mode 2, with target_rank = .1
        LR = 0.01
        lmbda = 1
        dim = (100, 200)
        target_rank = .1
        layers = 2
        for j in range(1):
            torch.manual_seed(j)

            model_A = self.BasicRectangularModel(dim[0], dim[1], layers=layers)

            for index in range(layers):
                init_matrix = torch.rand(dim)
                model_A.model[index].weight.data = init_matrix.clone()
            print(model_A)

            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Pre Model A Layer:{index}")

            new_q3r = QuaRS_v2(trainable_weights=[layer for layer in model_A.model], target_rank=target_rank,
                               lmbda=lmbda, steps=5,
                               rectangular_mode=2, verbose=True, scaling=1)

            self.assertEquals(new_q3r.target_ranks, [20, 20])


class WxSimilarityUnitTest(unittest.TestCase):
    def testEquivalentSquare(self):
        torch.manual_seed(0)
        LMBDA = 0.001
        A, B = torch.rand((100, 100)), torch.rand((100, 100))
        device = A.device

        wx_new = WxNew(target_rank=10, lmbda=LMBDA, rectangular_mode=False, device=device)
        wx_old = WxOld(target_rank=10, lmbda=LMBDA, rectangular_mode=False)
        wx_new.update_weightoperator(A, epsilon_large_flag=True), wx_old.update_weightoperator(A,
                                                                                               epsilon_large_flag=True)
        self.assertAlmostEqual(wx_new.val.item(), wx_old.val.item(), places=5,
                               msg=f"Update weight operator does not work")

        wx_new(A), wx_old(A)
        self.assertAlmostEqual(wx_new.val.item(), wx_old.val.item(), places=5,
                               msg=f"Components Diverge")
        for i in range(0, 100):
            wx_new(B), wx_old(B)
            self.assertAlmostEqual(wx_new.val.item(), wx_old.val.item(), places=5,
                                   msg=f"Components Diverge")

    def testEquivalentSquareWxV3(self):
        torch.manual_seed(0)
        LMBDA = 0.001
        A, B = torch.rand((100, 100)), torch.rand((100, 100))
        device = A.device

        wx_new = WxNewV2(A, target_rank=10, lmbda=LMBDA)
        wx_old = WxOld(target_rank=10, lmbda=LMBDA, rectangular_mode=False)

        wx_old.update_weightoperator(A, epsilon_large_flag=True)  # inits wx_old


        #self.assertAlmostEqual(wx_new.val.item(), wx_old.val.item(), places=5,
        #                       msg=f"Update weight operator does not work")

        print(wx_new.val.item(), wx_old.val.item())
        wx_new(A), wx_old(A)
        """self.assertAlmostEqual(wx_new.val.item(), wx_old.val.item(), places=5,
                               msg=f"Components Diverge")"""
        print(wx_new.val.item(), wx_old.val.item())

        for i in range(0, 100):
            wx_new(B), wx_old(B)
            print(wx_new.val.item(), wx_old.val.item())

            """self.assertAlmostEqual(wx_new.val.item(), wx_old.val.item(), places=5,
                                   msg=f"Components Diverge")"""

    def testEquivalentRectangular(self):
        torch.manual_seed(0)
        LMBDA = 0.001
        A, B = torch.rand((50, 100)), torch.rand((50, 100))
        device = A.device

        wx_new = WxNew(target_rank=10, lmbda=LMBDA, rectangular_mode=True, device=device)
        wx_old = WxOld(target_rank=10, lmbda=LMBDA, rectangular_mode=2)
        wx_new.update_weightoperator(A, epsilon_large_flag=True), wx_old.update_weightoperator(A,
                                                                                               epsilon_large_flag=True)
        self.assertAlmostEqual(wx_new.val.item(), wx_old.val.item(), places=5,
                               msg=f"Update weight operator does not work")

        wx_new(A), wx_old(A)
        self.assertAlmostEqual(wx_new.val.item(), wx_old.val.item(), places=5,
                               msg=f"Components Diverge iteration 1")
        for i in range(0, 100):
            wx_new(A), wx_old(A)
            self.assertAlmostEqual(wx_new.val.item(), wx_old.val.item(), places=5,
                                   msg=f"Components Diverge iteration {i + 2}")


if __name__ == '__main__':
    Q3R_unittest.main()
