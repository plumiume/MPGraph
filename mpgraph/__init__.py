# Copyright 2024 plumiume.com
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

from itertools import chain
from typing import Sequence
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS, FACEMESH_FACE_OVAL
import torch

def filter_conn(connection: Sequence[tuple[int, int]], indices: Sequence[int]):
    return [
        (a, b) for a, b in connection
        if a in indices and b in indices
    ]

def conn_to_index(connection: Sequence[tuple[int, int]], *connections: Sequence[tuple[int, int]]) -> torch.Tensor:
    """_summary_
    Returns:
        torch.Tensor: Tensor of (2, Edge)
    """
    connections = chain(connection, *connections)
    return torch.cat([
        torch.tensor(list(conns) + [(b, a) for a, b in conns])
        for conns in connections
    ]).transpose(1, 0)
