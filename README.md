# 🚀 Kubeflow Pipelines Examples

A playground of ML workflows built with Kubeflow Pipelines (KFP)! 🎮

## What is Kubeflow Pipelines? 🤔

Kubeflow Pipelines is a platform for building and deploying portable, scalable machine learning (ML) workflows based on Docker containers. The KFP SDK lets you define workflows using pure Python — no YAML needed! ✨

---

## 🧠 Core Concepts

| Concept | Description |
|---------|-------------|
| **Components** (`@dsl.component`) | 🧱 Single steps in a pipeline — like mini services that do one thing well. Each runs in its own Docker container. |
| **Pipelines** (`@dsl.pipeline`) | 🧩 DAGs that chain components together. Outputs from one step flow into the next! |
| **Compiler** | 🏭 Converts Python to YAML — the "blueprint" your Kubernetes cluster understands. |

---

## 📚 Examples

### 1️⃣ Basic Math Pipeline (`01_basic_math.py`) 🧮

A gentle intro to KFP — add numbers, multiply them, watch the data flow! 🌊

**What you'll learn:**
- ✅ Creating simple Python components
- ✅ Chaining components sequentially
- ✅ Passing data between steps
- ✅ Compiling to YAML

```bash
python 01_basic_math.py
# Generates: compiled_math_pipeline.yaml 📄
```

---

### 2️⃣ MNIST PyTorch Pipeline (`02_mnist.py`) 🧠

Train a CNN on handwritten digits! 🖊️ This pipeline is packed with features:

**What you'll learn:**
- ✅ Using custom base images + extra packages
- ✅ Structuring complex training logic
- ✅ Returning multiple outputs with `NamedTuple`
- ✅ GPU allocation & node selectors 🎮

```bash
python 02_mnist.py
# Generates: compiled_mnist_pipeline.yaml 📄
```

---

### 3️⃣ Volume Pipeline (`03_volumes.py`) 💾

Persist data to a Kubernetes Persistent Volume Claim (PVC)! 🗃️

**What you'll learn:**
- ✅ Mounting PVCs to containers 📥
- ✅ Writing files to persistent storage
- ✅ Resource configuration (CPU/Memory)

**Pipeline flow:** Input string → timestamped file on `/mnt/data` 📂

```bash
python 03_volumes.py
# Generates: compiled_volume_pipeline.yaml 📄
```

> 📌 Assumes a PVC named `tutorial` exists in your cluster.

---

### 4️⃣ Container Image Pipeline (`container/pipeline.py`) 🤖

**Run LLM inference with Qwen3.5-0.8B!** 🧠💬

**What you'll learn:**
- ✅ Using custom container images in pipelines 🐳
- ✅ Running scripts with command-line args
- ✅ GPU-powered inference 🚀
- ✅ Saving outputs to PVC

**Pipeline flow:** Prompt → Qwen model → generated text → `/mnt/data` 📝

**📁 Files:**
- `container/main.py` — Model inference script (with docstrings and CLI help)
- `container/Dockerfile` — Custom image with transformers
- `container/pipeline.py` — KFP pipeline definition

**🚀 Build & push:**
```bash
# First, log in to Docker
docker login

# Build and push (replace {user} with your Docker Hub username)
docker build -t {user}/kubeflow-tut container/ && docker push {user}/kubeflow-tut
```

**🔨 Compile pipeline:**
```bash
python container/pipeline.py
# Generates: compiled_qwen_pipeline.yaml 📄
```

> 📌 Requirements:
> - PVC named `tutorial` exists
> - GPU nodes with `nvidia.com/gpu` label
> - Container image accessible to your cluster

---

## 🎯 How to Use the Kubeflow UI

1. **Open** the Kubeflow Pipelines dashboard 🌐
2. **Upload** your `.yaml` file 📤
3. **Create** an Experiment 🧪
4. **Run** your pipeline! 🏃‍♂️

---

## 📝 Code Organization & Comments

All pipeline files follow a consistent structure with clear section markers:

| Section | Marker | Purpose |
|---------|--------|---------|
| **Imports** | `# ====` | File imports at the top |
| **Components** | `# 🧱` or `# 🤖` | Component definitions |
| **Pipeline** | `# 📐` | Pipeline definition with flow |
| **Compilation** | `# ⚙️` | Main compilation logic |

Each component includes:
- 📖 Clear docstrings with Args/Returns
- 💡 Inline comments for complex logic
- 🔍 Type hints for better IDE support

---

## 🎉 Happy Pipeline Building!
