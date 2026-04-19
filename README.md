# Distributed NeRF Renderer

The main objective is to design and implement a distributed rendering system for 3D scenes utilizing Neural Radiance Fields (NeRF) technology.

## Technical Focus & Scope

It is important to note that this project is not focused on AI research or the algorithmic implementation of NeRF. The neural network is treated as a computational "black-box" workload designed to stress the system.

The core technical merit and evaluation of this project lie in the distributed infrastructure:

- **Node Orchestration**: Managing multiple worker nodes in a local network cluster.
- **Polyglot Communication**: Integrating a highly concurrent Master node written in C++ with AI Workers running Python.
- **Data Parallelism**: Dynamically partitioning the image matrix (ray dispatching) across available processing nodes.
- **Fault Tolerance**: Implementing mechanisms to handle node timeouts, disconnections, or "straggler" mitigation during the rendering pipeline.

## Acknowledgments & References

This project is an architectural wrapper focused on Parallel and Distributed Systems (SPD). The underlying Deep Learning mathematical model and the datasets used for training are not original to this repository and are based on the pioneering work on **Neural Radiance Fields**.

* **Original Paper:** Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*. European Conference on Computer Vision (ECCV).
* **Core Neural Implementation:** The ray-marching logic and neural network inference scripts (`nerf_core/model.py`) were heavily adapted from the `tiny_nerf` implementation provided in the official [NeRF GitHub Repository](https://github.com/bmild/nerf).
* **Datasets:** All scene datasets used for training, were provided by the original authors.
