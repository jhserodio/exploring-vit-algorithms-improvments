# Usa a imagem oficial de desenvolvimento da NVIDIA com CUDA 12.4
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Evita prompts interativos durante a construção
ENV DEBIAN_FRONTEND=noninteractive

# Fixa a arquitetura da RTX 5070 Ti (Ada Lovelace) para o compilador PTX
ENV CUDA_COMPUTE_CAP=89
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Instala dependências do sistema, ferramentas de build, LLVM (para cudarc) e álgebra linear
RUN apt-get update && apt-get install -y \
  curl \
  git \
  build-essential \
  cmake \
  pkg-config \
  libssl-dev \
  clang \
  llvm-dev \
  libopenblas-dev \
  liblapack-dev \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

# Instala a toolchain do Rust para todos os usuários do contêiner
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH="/usr/local/cargo/bin:${PATH}"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
  && rustup component add rust-analyzer rustfmt

# Cria o diretório de trabalho principal
WORKDIR /workspace

# Comando padrão para manter o contêiner vivo em background
CMD ["tail", "-f", "/dev/null"]
