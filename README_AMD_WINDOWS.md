# SpargeAttn - AMD ROCm Setup Guide

This guide explains how to build and run SpargeAttn with AMD GPUs using ROCm on both Linux and Windows.

## Supported Hardware

SpargeAttn supports:
- **RDNA4 GPUs** (gfx1200, gfx1201) - RX 9060, RX 9070 series - **Full FP8 WMMA support**
- **RDNA3/RDNA3.5 GPUs** (gfx1100, gfx1101, gfx1102, gfx1103, gfx1151) - RX 7000 series - FP16 only

RDNA4 uses optimized FP8 kernels via rocWMMA for the S@V phase, providing better performance and memory efficiency. The kernel uses architecture-specific WMMA register layouts that are automatically selected at compile time.

---

## Linux Installation

### Prerequisites

- Linux (Ubuntu 22.04+ recommended)
- Python 3.10, 3.11, or 3.12
- ROCm 7.2+ (including rocWMMA)
- PyTorch with ROCm support

### 1. Install ROCm

Follow the official [ROCm installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) for your Linux distribution.

For Ubuntu:
```bash
# Add ROCm repository
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.4/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Install ROCm
sudo apt update
sudo apt install rocm-dev rocm-libs rocwmma-dev
```

### 2. Install PyTorch with ROCm

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Or for nightly builds with newer ROCm:
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3
```

For RDNA4 GPUs, you may need nightly wheels from TheRock:
```bash
# For gfx120X (RX 9060, RX 9070)
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ --pre torch torchaudio torchvision
```

### 3. Install Triton

```bash
pip install triton
```

### 4. Build and Install SpargeAttn

```bash
# Clone the repository
git clone https://github.com/your-repo/SpargeAttn.git
cd SpargeAttn

# Set environment variables
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH

# Install in editable mode (recommended for development)
pip install --no-build-isolation -e .

# Or standard install
pip install --no-build-isolation .
```

---

## Windows Installation

### Prerequisites

- Windows 10/11
- Python 3.11, 3.12, or 3.13
- Visual Studio 2022 with C++ build tools
- AMD Adrenaline driver (latest recommended)

### 1. Install ROCm and PyTorch from TheRock

Follow the instructions at [ROCm/TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) to install ROCm and PyTorch wheels for your GPU architecture.

#### Create a Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Install PyTorch (includes ROCm SDK as dependency)

For **gfx1151** (AMD Strix Halo iGPU):
```powershell
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision
```

For **gfx110X** (RX 7900 XTX, RX 7800 XT, RX 7700S, Radeon 780M):
```powershell
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ --pre torch torchaudio torchvision
```

For **gfx120X** (RX 9060, RX 9070):
```powershell
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ --pre torch torchaudio torchvision
```

#### Initialize ROCm SDK

```powershell
rocm-sdk init
```

#### Install Triton with AMD Windows Support

```powershell
pip install triton-windows
```

### 2. Set Environment Variables

Open a PowerShell terminal and run:

```powershell
# Activate Visual Studio environment
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && set' | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Set ROCm paths using rocm-sdk
$ROCM_ROOT = (rocm-sdk path --root).Trim()
$ROCM_BIN = (rocm-sdk path --bin).Trim()
$env:ROCM_HOME = $ROCM_ROOT
$env:PATH = "$ROCM_ROOT\lib\llvm\bin;$ROCM_BIN;$env:PATH"

# Set compiler and build settings
$env:CC = "clang-cl"
$env:CXX = "clang-cl"
$env:DISTUTILS_USE_SDK = "1"

# Enable experimental features
$env:FLASH_ATTENTION_TRITON_AMD_ENABLE = "TRUE"
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"
```

### 3. Build and Install SpargeAttn

```powershell
cd <path_to_spargeattn>
pip install --no-build-isolation -e .
```

---

## Testing

### Quick Correctness Test

Run this script to verify SpargeAttn is working correctly by comparing against PyTorch SDPA:

```python
import torch
import torch.nn.functional as F
from spas_sage_attn import spas_sage_attn_meansim_cuda

device = torch.device('cuda')

# Create random test tensors
q = torch.randn(1, 12, 2048, 128, dtype=torch.float16, device=device)
k = torch.randn(1, 12, 2048, 128, dtype=torch.float16, device=device)
v = torch.randn(1, 12, 2048, 128, dtype=torch.float16, device=device)

# Compute reference output using PyTorch SDPA
with torch.no_grad():
    sdpa = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).to(torch.float16)

# Compute SpargeAttn output (with 100% density = no sparsity)
sparge = spas_sage_attn_meansim_cuda(
    q, k, v,
    is_causal=False,
    smooth_k=False,
    simthreshd1=0.0,   # No similarity threshold (keep all blocks)
    cdfthreshd=1.0,    # 100% density
    pvthreshd=0,
    tensor_layout='HND'
)

# Compare outputs using cosine similarity
cos = F.cosine_similarity(
    sdpa.flatten().float().unsqueeze(0),
    sparge.flatten().float().unsqueeze(0)
)
print(f'Cosine similarity: {cos.item():.6f}')  # Should be > 0.999
```

Save as `test_spargeattn.py` and run:
```bash
python test_spargeattn.py
```

Expected output:
```
Cosine similarity: 0.999xxx
```

### Testing FP8 on RDNA4

On RDNA4 GPUs, you can use the FP8-optimized kernel:

```python
from spas_sage_attn import spas_sage2_attn_meansim_cuda

# Use the FP8 variant (requires RDNA4 or MI series GPU)
sparge_fp8 = spas_sage2_attn_meansim_cuda(
    q, k, v,
    is_causal=False,
    smooth_k=False,
    simthreshd1=0.0,
    cdfthreshd=1.0,
    pvthreshd=0,
    tensor_layout='HND'
)
```

---

## Performance Notes

At L=4096, D=128, bf16 vs PyTorch SDPA (with aotriton):

| Sparsity | Time | Speedup vs SDPA |
|----------|------|-----------------|
| 100% | 33.0 ms | 0.18x |
| 50% | 13.7 ms | 0.43x |
| 25% | 7.4 ms | 0.79x |
| **10%** | **3.2 ms** | **1.81x** |
| 5% | 1.8 ms | 3.26x |
| 2% | 1.0 ms | 6.07x |

**Break-even point**: ~20-25% sparsity. Below that, SpargeAttn is faster than dense SDPA.

---

## Known Issues

1. **No FP8 support on RDNA3** - RDNA3 (gfx10xx/gfx11xx) does not support FP8 WMMA at the hardware level, so FP16 kernels are used. RDNA4 (gfx12xx) fully supports FP8 WMMA.

2. **Triton compiler warnings** - You may see `clang-cl: warning: unknown argument ignored` warnings during first run. These are harmless.

3. **Architecture-specific WMMA layouts** - The kernels use different WMMA register layouts for gfx11 (RDNA3) vs gfx12 (RDNA4). This is handled automatically at compile time.

---

## Troubleshooting

### Linux

#### "hipErrorNoBinaryForGpu" or similar GPU errors
Make sure your ROCm installation matches your GPU architecture. Check with:
```bash
rocminfo | grep gfx
```

#### Build fails with missing rocWMMA
Install rocWMMA development package:
```bash
sudo apt install rocwmma-dev
```

### Windows

#### "LoadLibrary failed" or "cannot find amdhip64.dll"
Make sure you ran `rocm-sdk init` after installing the ROCm SDK packages.

#### "LINK : fatal error LNK1104: cannot open file 'python312.lib'"
Ensure Visual Studio environment is activated before building:
```powershell
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && set' | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }
```

#### "PermissionError" when compiling Triton kernels
This is a known Windows issue with temp file handling. Make sure you're using the latest `triton-windows` package:
```powershell
pip install --upgrade triton-windows
```
