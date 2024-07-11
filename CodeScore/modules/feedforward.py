# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Feed Forward
============
    Feed Forward Neural Network module that can be used for classification or regression
"""
from typing import List, Optional

import torch
from torch import nn
import copy


class FeedForward(nn.Module):
    """Feed Forward Neural Network.

    Args:
        in_dim (int): Number input features.
        out_dim (int): Number of output features. Default is just a score.
        hidden_sizes (List[int]): List with hidden layer sizes. Defaults to [3072,1024]
        activations (str): Name of the activation function to be used in the hidden
            layers. Defaults to 'Tanh'.
        final_activation (Optional[str]): Final activation if any.
        dropout (float): dropout to be used in the hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_sizes: List[int] = [3072, 1024],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        modules = []
        modules.append(nn.Linear(in_dim, hidden_sizes[0]))
        modules.append(self.build_activation(activations))
        modules.append(nn.Dropout(dropout))

        for i in range(1, len(hidden_sizes)):
            modules.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            modules.append(self.build_activation(activations))
            modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(hidden_sizes[-1], int(out_dim)))
        if final_activation is not None:
            modules.append(self.build_activation(final_activation))

        self.ff = nn.Sequential(*modules)

    def build_activation(self, activation: str) -> nn.Module:
        if hasattr(nn, activation.title()):
            return getattr(nn, activation.title())()
        else:
            raise Exception(f"{activation} is not a valid activation function!")

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return self.ff(in_features)


class FeedForward_exec(nn.Module):
    """Feed Forward Neural Network.

    Args:
        in_dim (int): Number input features.
        out_dim (int): Number of output features. Default is just a score.
        hidden_sizes (List[int]): List with hidden layer sizes. Defaults to [3072,1024]
        activations (str): Name of the activation function to be used in the hidden
            layers. Defaults to 'Tanh'.
        final_activation (Optional[str]): Final activation if any.
        dropout (float): dropout to be used in the hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 3, # score, exec, pass@1
        hidden_sizes: List[int] = [3072, 1024],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        modules = []
        modules.append(nn.Linear(in_dim, hidden_sizes[0]))
        modules.append(self.build_activation(activations))
        modules.append(nn.Dropout(dropout))

        for i in range(1, len(hidden_sizes)):
            modules.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            modules.append(self.build_activation(activations))
            modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(hidden_sizes[-1], int(out_dim)))
        self.ff2 = nn.Sequential(*copy.deepcopy(modules))
        self.ff3 = nn.Sequential(*copy.deepcopy(modules))

        if final_activation is not None:
            modules.append(self.build_activation(final_activation))


        self.ff1 = nn.Sequential(*copy.deepcopy(modules))

    def build_activation(self, activation: str) -> nn.Module:
        if hasattr(nn, activation.title()):
            return getattr(nn, activation.title())()
        else:
            raise Exception(f"{activation} is not a valid activation function!")

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        score_output = self.ff1(in_features)
        passed_output = self.ff2(in_features)
        pass_at_1_output = self.ff3(in_features)
        return torch.cat((score_output, passed_output, pass_at_1_output), dim=1)
        # return self.ff2(in_features)

