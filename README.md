# 🚀 Tempest

[![Build Passing](https://github.com/ashfaq1701/tempest/actions/workflows/cpu-tests.yml/badge.svg?branch=master)](https://github.com/ashfaq1701/tempest/actions/workflows/cpu-tests.yml)
[![PyPI Latest Release](https://img.shields.io/pypi/v/tempest-rw.svg)](https://pypi.org/project/tempest-rw/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/tempest-rw.svg)](https://pypi.org/project/tempest-rw/)

**A high-performance temporal random walk sampler for dynamic networks with GPU acceleration. Built for scale.**

---

## 🔥 Why Tempest?
✅ **Performance First** – GPU-accelerated sampling for massive networks<br>
✅ **Memory Efficient** – Smart memory management for large graphs  
✅ **Flexible Integration** – Easy Python bindings with **NumPy/NetworkX** support  
✅ **Production Ready** – Tested with hundreds of extensive unit tests.<br>
✅ **Multi Platform** Builds and runs seamlessly on devices with or without CUDA.

---

## ⚡ Quick Start

```python
from tempest import Tempest

# Create a directed temporal graph
walker = Tempest(is_directed=True, use_gpu=True, max_time_capacity=-1)

# Add edges - can be numpy arrays or python lists
sources = [3, 2, 0, 3, 3, 1]
targets = [4, 4, 2, 1, 2, 4]
timestamps = [71, 82, 19, 34, 79, 19]

walker.add_multiple_edges(sources, targets, timestamps)

# Sample walks with exponential time bias
walk_nodes, walk_timestamps, walk_lens, edge_features = walker.get_random_walks_and_times_for_all_nodes(
    max_walk_len=5,
    walk_bias="ExponentialIndex",
    num_walks_per_node=10,
    initial_edge_bias="Uniform"
)
# edge_features is None when no edge features were added (feature_dim=0)
```

## ✨ Key Features
- ⚡ **GPU acceleration** for large graphs   
- 🎯 **Multiple sampling strategies** – Uniform, Linear, Exponential
- 🧠 **Advanced temporal biases** – ExponentialWeight (CTDNE-style) and TemporalNode2Vec
- 🔄 **Forward & backward** temporal walks
- 📡 **Rolling window support** for streaming data
- 🏷️ **Optional edge feature propagation** from input edges to sampled walks
- 🔗 **NetworkX integration**
- 🛠️ **Efficient memory management**
- ⚙️ Uses **C++ std libraries** or **Thrust API** selectively based on hardware availability and configuration. 

---

## 🏷️ Edge Features (Optional)

If your edges carry attributes (weights, embeddings, types, etc.), you can pass them to
`add_multiple_edges(...)` and receive aligned edge features for each sampled transition.

```python
import numpy as np
from tempest import Tempest

walker = Tempest(is_directed=True, use_gpu=False)

sources = np.array([0, 0, 1], dtype=np.int32)
targets = np.array([1, 2, 2], dtype=np.int32)
timestamps = np.array([10, 20, 30], dtype=np.int64)

# shape: [num_edges, feature_dim]
edge_features = np.array([
    [0.1, 1.0],
    [0.2, 0.5],
    [0.9, 0.3],
], dtype=np.float32)

walker.add_multiple_edges(sources, targets, timestamps, edge_features=edge_features)

walk_nodes, walk_timestamps, walk_lens, walk_edge_features = walker.get_random_walks_and_times(
    max_walk_len=4,
    walk_bias="Uniform",
    num_walks_total=5,
)

# walk_edge_features.shape == [num_walks, max_walk_len - 1, feature_dim]
```

`walk_edge_features` is `None` when no edge features are provided.

## 🏷️ Node Features

The library can also store dense node features. Use `set_node_features(node_ids, node_features)`
to populate features for specific nodes, then `get_node_features()` to retrieve the dense matrix.

---

## 🧭 Bias Selection Notes

- Use `ExponentialIndex` or `Linear` for recency-aware sampling with no extra setup.
- Use `ExponentialWeight` when you want CTDNE-style weight computation
  (`enable_weight_computation=True`, optionally tune `timescale_bound`).
- Use `TemporalNode2Vec` when you need return/in-out control via
  `temporal_node2vec_p` and `temporal_node2vec_q`.

---

## 📦 Dependencies

| Dependency     | Purpose                                     |
|---------------|---------------------------------------------|
| **pybind11**  | Python-C++ bindings                         |
| **python3**   | Required for building the python interfaces |
| **gtest**     | Unit testing framework                      |


> 💡 **Tip:** Use **vcpkg** to easily install and link the C++ dependencies.

---

## 📦 Installation

### GPU (default) — Linux with NVIDIA driver

```sh
pip install tempest-rw
```

This pulls the manylinux wheel plus the two NVIDIA CUDA runtime
libraries that the wheel links against:

- `nvidia-cuda-runtime-cu12` — provides `libcudart.so.12`
- `nvidia-curand-cu12` — provides `libcurand.so.10`

### CPU-only — source build

Machines without a GPU (or without a driver) should build from
source; the CMake config detects the absence of `nvcc` and compiles
a CPU-only extension:

```sh
pip install --no-binary tempest-rw tempest-rw
```

You'll need a C++17 compiler, CMake, and OpenMP/TBB installed
(via `vcpkg` — see the Dependencies section above).

## 📖 Documentation

📌 **[C++ Documentation →](https://htmlpreview.github.io/?https://github.com/ashfaq1701/tempest/blob/master/docs/html/Tempest_8cuh_source.html)**<br>
📌 **[Python Interface Documentation →](docs/_tempest.md)**

---

## 📚 Citation

If you use Tempest in your research, please cite:

> Md Ashfaq Salehin, George Parisis, and Luc Berthouze. 2026. **Tempest: A GPU-Accelerated Engine for Streaming Temporal Random Walks**. In *Proceedings of the 55th International Conference on Parallel Processing (ICPP '26)*, Singapore.

```bibtex
@inproceedings{salehin2026tempest,
  title     = {Tempest: A GPU-Accelerated Engine for Streaming Temporal Random Walks},
  author    = {Salehin, Md Ashfaq and Parisis, George and Berthouze, Luc},
  booktitle = {Proceedings of the 55th International Conference on Parallel Processing (ICPP '26)},
  address   = {Singapore},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/XXXXXXX.XXXXXXX}
}
```

---

## 👨‍🔬 Built by [Packets Research Lab](https://packets-lab.github.io/)

🚀 **Contributions welcome!** Open a PR or issue if you have suggestions.  
