# 🔬 ViT Inference Engine: Tumoral Region Recognition (OSCC)

This repository contains the engineering infrastructure and inference engine for the scientific research project focused on the algorithmic optimization of Vision Transformers (ViT) using linear algebra and numerical calculus to reduce computational hardware costs. The primary goal is the recognition of tumoral regions in oral histological images (Oral Squamous Cell Carcinoma - OSCC).

The architecture is designed to extract maximum parallel processing performance, utilizing a strictly containerized Rust environment to bypass Python GIL bottlenecks and achieve granular VRAM management.

## 💻 Environment Specifications (Hardware & OS)

The artificial intelligence stack was designed and validated on the following physical hardware:

* **OS:** Fedora Linux 41 (KDE Plasma)
* **CPU:** AMD Ryzen 7 7700XD (Data parallelism managed via physical/logical threads)
* **GPU:** NVIDIA RTX 5070 Ti (CUDA Compute Capability: 89)
* **Storage:** 2 TB NVMe SSD (I/O optimized for gigapixel matrix tiling)

---

## ⚙️ Prerequisites and Host Configuration

To ensure exact reproducibility and prevent memory allocation failures in the Linux kernel when manipulating tensors with billions of parameters, the host machine must be configured according to the script below before initializing Docker Compose.

### 1. Linux Kernel Tuning
Increases the virtual memory mapping limits and the maximum number of open files.

```bash
# Immediate application
sudo sysctl -w vm.max_map_count=262144
sudo sysctl -w fs.file-max=6553500

# Persistence after reboot
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.d/99-ai-tuning.conf
echo "fs.file-max=6553500" | sudo tee -a /etc/sysctl.d/99-ai-tuning.conf
```

### 2. Docker Engine Installation (DNF5)
On Fedora 41, due to the transition to the DNF5 package manager, the official repository is injected via direct download.

```bash

# Remove legacy stack and conflicting tools (Podman/Moby)
sudo dnf remove docker docker-client docker-client-latest docker-common docker-latest docker-latest-logrotate docker-logrotate docker-selinux docker-engine-selinux docker-engine podman runc

# Dependencies and official repository injection
sudo dnf -y install dnf-plugins-core
sudo wget -O /etc/yum.repos.d/docker-ce.repo [https://download.docker.com/linux/fedora/docker-ce.repo](https://download.docker.com/linux/fedora/docker-ce.repo)

# Install the main engine and orchestration plugin
sudo dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 3. NVIDIA Container Toolkit Integration
The low-level communication bridge that allows containers to map the local GPU without virtualization overhead.
Bash

```bash
# Add the toolkit repository
curl -s -L [https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo](https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo) | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install the base package
sudo dnf install -y nvidia-container-toolkit
```

### 4. Docker Daemon Parameterization
Sets NVIDIA as the default runtime for all ML containers and configures the log driver to prevent exhausting the NVMe storage.
Bash

```bash
sudo mkdir -p /etc/docker
cat <<EOF | sudo tee /etc/docker/daemon.json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  }
}
EOF
```

### 4. Model Weights Management (Hugging Face CLI)

To avoid bloating the Rust inference engine with Python dependencies, model weights are managed directly on the host machine using an isolated virtual environment. The downloaded tensors are then mapped into the Docker container as read-only volumes.

**Environment Setup & Activation**
```bash
# Create the virtual environment on the host machine (Fedora)
python3 -m venv ~/.env_ai

# Activate the environment
source ~/.env_ai/bin/activate

# Install the Hugging Face CLI
pip install -U "huggingface_hub[cli]"

```

**Model Management Commands Guide**
Once the environment is active (`(.env_ai)` prompt), use the CLI to download and manage the Foundation Models. By default, weights are cached in `~/.cache/huggingface/hub/`.

-   **Download DINOv2 (Base Model):**
    Downloads the complete repository structure and weights.

    ```bash
    huggingface-cli download facebook/dinov2-base

    ```

-   **Download Specific Tensor Formats:**
    When optimizing for Rust/Candle, it is often more efficient to download only the `safetensors` format, avoiding heavy PyTorch `.bin` files.

    ```bash
    huggingface-cli download facebook/dinov2-base model.safetensors

    ```

-   **Scan Local Cache:**
    Monitor how much space the models are consuming on your NVMe drive.

    ```bash
    huggingface-cli scan-cache

    ```

-   **Delete Unused Weights:**
    Interactively select and remove old or unused model revisions to free up storage.


    ```bash
    huggingface-cli delete-cache

    ```

### 5. Bridging the Host Cache to the Rust Engine

To ensure the compiled Rust binary inside the Docker container can load the models without redownloading them, update the `volumes` section of your `docker-compose.yml` to mount the Hugging Face cache in read-only mode (`:ro`).

```YAML

```
    volumes:
      - .:/workspace
      - /path/to/fedora/ssd/images:/data
      # Mount the local model cache into the container
      - ~/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro
```
```
```
