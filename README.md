# Kubeflow Pipelines Examples

This repository contains examples of how to build and compile machine learning pipelines using the Kubeflow Pipelines (KFP) SDK.

## What is Kubeflow Pipelines?

Kubeflow Pipelines is a platform for building and deploying portable, scalable machine learning (ML) workflows based on Docker containers. The KFP SDK allows you to specify your workflows using pure Python.

## Core Concepts

1. **Components (`@dsl.component`)**: A component is a single step in a pipeline. It represents a self-contained piece of code that does one thing (e.g., data preprocessing, model training). Each component runs in its own Docker container. Because they are isolated, all imports and helper functions must be defined *inside* the component function.
2. **Pipelines (`@dsl.pipeline`)**: A pipeline is a Directed Acyclic Graph (DAG) that connects multiple components together. You define the flow of execution by passing the outputs of one component as inputs to the next.
3. **Compiler**: The KFP compiler takes your Python pipeline definition and compiles it into a YAML file (Intermediate Representation). This YAML file can then be uploaded to the Kubeflow Pipelines UI to be executed on a Kubernetes cluster.

## Examples Provided

### 1. Basic Math Pipeline (`01_basic_math.py`)

A simple introductory pipeline that demonstrates:
- How to create basic components using pure Python functions.
- How to define a pipeline that links these components sequentially.
- How to pass data between steps (e.g., passing the output of the addition task to the multiplication task).
- How to compile the pipeline to a `.yaml` file.

**Usage:**
```bash
python 01_basic_math.py
```
This generating `compiled_math_pipeline.yaml` can be uploaded to the Kubeflow UI.

### 2. MNIST PyTorch Pipeline (`02_mnist.py`)

A more advanced pipeline that trains a Convolutional Neural Network (CNN) on the MNIST dataset using PyTorch. This example demonstrates:
- Specifying base images and requesting additional packages via `packages_to_install`.
- Structuring complex components (putting imports, classes, and execution logic *inside* the function).
- Returning multiple output values from a component using `NamedTuple`.
- Configuring resource limits (CPU, Memory).
- Requesting hardware accelerators (GPUs) and setting node constraints for scheduling (`nvidia.com/gpu`).

**Usage:**
```bash
python 02_mnist.py
```
This generates `compiled_mnist_pipeline.yaml` which can be uploaded to your Kubeflow cluster.

### 3. Volume Pipeline (`03_volumes.py`)

A pipeline that demonstrates how to persist data to a Kubernetes Persistent Volume Claim (PVC). This example shows:
- Mounting a PVC to a task container using `kubernetes.mount_pvc()`.
- Writing files to the mounted volume path within the component.
- Configuring CPU and memory resource requests and limits.

The pipeline takes a test string as input, writes it to a timestamped file on the mounted volume, and saves it to `/mnt/data`.

**Usage:**
```bash
python 03_volumes.py
```
This generates `compiled_volume_pipeline.yaml` which can be uploaded to your Kubeflow cluster. The pipeline assumes a PVC named `tutorial` exists in your Kubernetes cluster.

### 4. Container Image Pipeline (`container/pipeline.py`)

A pipeline that uses a custom container image (`mbodenham/kubeflow-tut`) to run inference with the Qwen3.5-0.8B language model. This example demonstrates:
- Using a pre-built custom Docker image as the base image for a component.
- Running a Python script inside the container with command-line arguments.
- Mounting a PVC to save generated outputs to persistent storage.
- Configuring GPU resources for inference workloads.

The pipeline takes a text prompt as input, runs it through the Qwen3.5-0.8B model, and saves the generated response to a timestamped file on the mounted volume (`/mnt/data`).

**Files:**
- `container/main.py` - The Python script that loads the model and generates text
- `container/Dockerfile` - Builds the custom image with transformers and main.py
- `container/pipeline.py` - The Kubeflow pipeline definition

**Build and push the container image:**

First, log in to Docker Hub (or your preferred container registry):
```bash
docker login
```

Then build and push the image (replace `{user}` with your Docker Hub username):
```bash
docker build -t {user}/kubeflow-tut container/ && docker push {user}/kubeflow-tut
```

**Compile the pipeline:**
```bash
python container/pipeline.py
```

This generates `compiled_qwen_pipeline.yaml` which can be uploaded to your Kubeflow cluster. The pipeline assumes:
- A PVC named `tutorial` exists in your Kubernetes cluster
- The container image `{user}/kubeflow-tut` is pushed to a registry accessible by your cluster (update `base_image` in `container/pipeline.py` if using a different registry)
- Your cluster has GPU nodes with the `nvidia.com/gpu` label


## How to use Kubeflow UI
1. Open the Kubeflow Pipelines dashboard in your browser.
2. Navigate to the **Pipelines** section.
3. Click **Upload Pipeline** and select the `.yaml` file generated by the python scripts.
4. Once uploaded, you can create a **Experiment** then create a **Run** to execute the pipeline on your Kubernetes cluster.
