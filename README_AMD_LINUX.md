# SpargeAttn - AMD ROCm on Linux Setup Guide

This guide explains how to build and run SpargeAttn on Linux with AMD GPUs using ROCm.

## Supported Hardware

SpargeAttn on Linux has been tested with:
- **RDNA3** GPUs (gfx1100, gfx1101, gfx1102) - RX 7900 XTX, RX 7900 XT, RX 7800 XT, etc.
- **MI series** GPUs (gfx90a, gfx942) - MI200, MI300 series

### Feature Support by Architecture

| Architecture | INT8 QK | FP16 V | FP8 V | Notes |
|-------------|---------|--------|-------|-------|
| gfx1100/gfx11xx (RDNA3) | Yes | Yes | No | Use `spas_sage_attn_*` functions |
| gfx90a/gfx942 (MI series) | Yes | Yes | Yes | Use `spas_sage2_attn_*` functions |

## Prerequisites

- Linux (Ubuntu 22.04+ recommended)
- Python 3.9, 3.10, or 3.11
- ROCm 6.0+ (tested with ROCm 7.2)
- AMD GPU with ROCm support

## Installation

### 1. Install ROCm

Follow the official AMD ROCm installation guide for your Linux distribution:
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/

Verify ROCm is installed:
```bash
rocm-smi
# Should show your GPU(s)
```

### 2. Install PyTorch with ROCm Support

#### Option A: Official PyTorch ROCm wheels
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

#### Option B: Build from source or use nightly
For the latest ROCm support, you may need nightly builds:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
```

### 3. Install Triton

```bash
pip install triton
```

### 4. Set Environment Variables

```bash
# Set ROCm home (adjust path if ROCm is installed elsewhere)
export ROCM_HOME=/opt/rocm

# Optional: Set specific GPU architecture
export ROCM_ARCH=gfx1100  # Replace with your GPU arch

# For debugging (optional)
export SA_DEBUG=0  # Set to 1 for debug builds
```

### 5. Build and Install SpargeAttn

```bash
cd <path_to_spargeattn>

# Install in editable mode (recommended)
pip install --no-build-isolation -e .
```

> **Note:** Editable mode (`-e`) is recommended when working from the source directory. This avoids Python path shadowing issues where the source directory without compiled extensions could take precedence over the installed package.

#### Build Flags

The setup.py automatically detects your GPU architecture. To override:

```bash
# Force specific architecture
ROCM_ARCH=gfx1100 pip install --no-build-isolation -v .

# Debug build (unoptimized, with symbols)
SA_DEBUG=1 pip install --no-build-isolation -v .

# Custom rocWMMA path (if not using system ROCm)
ROCWMMA_INCLUDE_PATH=/path/to/rocwmma/include pip install --no-build-isolation -v .
```

## Testing

### Quick Kernel Test

```python
import torch
import torch.nn.functional as F
from spas_sage_attn.core import spas_sage_attn_meansim_cuda

device = torch.device('cuda')

# Create random test tensors
q = torch.randn(1, 12, 2048, 128, dtype=torch.bfloat16, device=device)
k = torch.randn(1, 12, 2048, 128, dtype=torch.bfloat16, device=device)
v = torch.randn(1, 12, 2048, 128, dtype=torch.bfloat16, device=device)

# Compute reference output using PyTorch SDPA
with torch.no_grad():
    sdpa = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).to(torch.bfloat16)

# Compute SpargeAttn output (with 100% sparsity = dense attention)
sparge = spas_sage_attn_meansim_cuda(
    q, k, v,
    is_causal=False,
    smooth_k=False,
    simthreshd1=0.0,   # No similarity threshold (keep all blocks)
    cdfthreshd=1.0,    # 100% sparsity
    pvthreshd=0,
    tensor_layout='HND'
)

# Compare outputs using cosine similarity
cos = F.cosine_similarity(
    sdpa.flatten().float().unsqueeze(0),
    sparge.flatten().float().unsqueeze(0)
)
print(f'Cosine similarity: {cos.item():.6f}')  # Should be ~0.9999
```

Save as `test_spargeattn.py` and run:
```bash
python test_spargeattn.py
```

Expected output:
```
Cosine similarity: 0.999900
```

### Running Inference Examples

#### CogVideoX Example

```bash
cd <path_to_spargeattn>

# Generate a video with 10% sparsity (topk=0.1)
python inference_examples/cogvideox_infer.py --mode topk --value 0.1 --start 0 --end 1
```

## API Usage

### For RDNA GPUs (gfx10xx, gfx11xx) - Use FP16 functions

```python
from spas_sage_attn import spas_sage_attn_meansim_cuda, spas_sage_attn_meansim_topk_cuda

# With CDF threshold
output = spas_sage_attn_meansim_cuda(q, k, v, cdfthreshd=0.9)

# With top-k ratio
output = spas_sage_attn_meansim_topk_cuda(q, k, v, topk=0.1)
```

### For MI series GPUs (gfx90a, gfx942) - Use FP8 functions

```python
from spas_sage_attn import spas_sage2_attn_meansim_cuda, spas_sage2_attn_meansim_topk_cuda

# With CDF threshold
output = spas_sage2_attn_meansim_cuda(q, k, v, cdfthreshd=0.9)

# With top-k ratio
output = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=0.1)
```

### Auto-detecting GPU Architecture

```python
import torch

def get_sparge_functions():
    """Returns appropriate SpargeAttn functions for current GPU."""
    if torch.version.hip is not None and torch.cuda.is_available():
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
        if arch.startswith('gfx10') or arch.startswith('gfx11'):
            # RDNA GPUs - use FP16 variant
            from spas_sage_attn import spas_sage_attn_meansim_cuda, spas_sage_attn_meansim_topk_cuda
            return spas_sage_attn_meansim_cuda, spas_sage_attn_meansim_topk_cuda

    # MI series or NVIDIA - use FP8 variant
    from spas_sage_attn import spas_sage2_attn_meansim_cuda, spas_sage2_attn_meansim_topk_cuda
    return spas_sage2_attn_meansim_cuda, spas_sage2_attn_meansim_topk_cuda
```

## Known Issues

1. **No FP8 support on RDNA3** - rocWMMA on gfx11xx doesn't support FP8, so FP16/BF16 is used for V. Use `spas_sage_attn_*` functions instead of `spas_sage2_attn_*`.

2. **Triton JIT compilation** - First run may be slow due to Triton kernel compilation. Subsequent runs use cached kernels.

3. **hipBLASLt warning** - You may see "Attempting to use hipBLASLt on an unsupported architecture" warnings on RDNA GPUs. This is harmless and can be ignored.

## Troubleshooting

### "rocm-smi not found" or "HIP not available"

Ensure ROCm is properly installed and in your PATH:
```bash
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
```

### Build fails with "rocwmma.hpp not found"

rocWMMA headers are needed. Either:
1. Install from system ROCm (usually at `/opt/rocm/include/rocwmma`)
2. Let setup.py auto-clone from GitHub
3. Set custom path: `ROCWMMA_INCLUDE_PATH=/path/to/rocwmma pip install ...`

### "undefined symbol" errors at runtime

Rebuild with matching ROCm version:
```bash
pip uninstall spas_sage_attn
pip install --no-build-isolation -v .
```

### GPU memory errors during inference

Try reducing batch size or sequence length. For very long sequences (>32K), ensure sufficient VRAM.

## Docker Example

```dockerfile
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

# Install dependencies
RUN pip install triton einops

# Clone and install SpargeAttn
WORKDIR /workspace
RUN git clone https://github.com/thu-ml/SpargeAttn.git
WORKDIR /workspace/SpargeAttn
RUN pip install --no-build-isolation -v .
```

Run with:
```bash
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /path/to/models:/models \
    spargeattn-rocm python test_spargeattn.py
```
