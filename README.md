# Hybrid AI-Quantum Satellite Image Encryption System

A secure satellite image encryption system that combines **AI-based semantic segmentation** (FlexiMo Vision Transformer) with **quantum encryption** (NEQR via Qiskit) to create a dual-layer security system.

## Architecture

```
Input Image
    │
    ▼
┌─────────────────────────┐
│  FlexiMo AI Segmentation │ ── Repository A (DOFA ViT)
│  (Vision Transformer)     │
└────────┬────────┬────────┘
         │        │
    ROI Mask   Background Mask
         │        │
         ▼        ▼
┌────────────┐  ┌──────────────┐
│ 8×8 Block  │  │ AES-256-GCM  │
│ Division   │  │ Classical     │
└─────┬──────┘  │ Encryption   │
      │         └──────┬───────┘
      ▼                │
┌────────────┐         │
│ NEQR Quantum│ ── Repository B (Qiskit AerSimulator)
│ Encryption  │
└─────┬──────┘         │
      │                │
      ▼                ▼
┌─────────────────────────┐
│    Fusion Engine         │
│  (Combine Components)    │
└────────┬────────────────┘
         │
         ▼
   Encrypted Image + Metadata
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Pretrained Weights
The DOFA ViT weights will be downloaded automatically on first run, or manually:
```bash
# Download from HuggingFace
wget https://huggingface.co/earthflow/DOFA/resolve/main/DOFA_ViT_base_e100.pth
```

### 3. Place Input Images
```bash
cp your_satellite_image.png input/
```

### 4. Run Encryption
```bash
python main.py --mode encrypt --input input/satellite.png
```

### 5. Run Decryption
```bash
python main.py --mode decrypt --original input/satellite.png
```

### 6. Verify Zero Data Loss
```bash
python main.py --mode verify --original input/satellite.png --decrypted output/decrypted/decrypted_satellite.png
```

## Modes

| Mode | Command | Description |
|------|---------|-------------|
| **encrypt** | `python main.py --mode encrypt` | Full encryption pipeline |
| **decrypt** | `python main.py --mode decrypt` | Full decryption pipeline |
| **analyze** | `python main.py --mode analyze` | ROI analysis only (no encryption) |
| **verify** | `python main.py --mode verify` | Compare original vs decrypted |

## Project Structure

```
quantum-image-encryption/
├── main.py                     # Master orchestrator
├── config/
│   └── config.json             # All settings
├── engines/
│   ├── ai_engine.py            # FlexiMo integration
│   ├── decision_engine.py      # ROI blocking
│   ├── quantum_engine.py       # NEQR encryption (Repo B)
│   ├── classical_engine.py     # AES encryption
│   ├── fusion_engine.py        # Combine encrypted parts
│   └── verification_engine.py  # Zero-loss verification
├── workflows/
│   ├── encrypt_workflow.py     # Encryption pipeline
│   ├── decrypt_workflow.py     # Decryption pipeline
│   ├── analyze_workflow.py     # ROI analysis only
│   └── verify_workflow.py      # Verification only
├── utils/
│   ├── image_utils.py          # Image operations
│   ├── block_utils.py          # 8×8 blocking
│   ├── crypto_utils.py         # Key generation
│   └── logger.py               # Logging
├── repos/                      # External repositories
│   ├── fleximo_repo/            # Repo A: FlexiMo AI
│   └── quantum_repo/           # Repo B: Quantum encryption
├── input/                      # Input satellite images
├── output/
│   ├── encrypted/              # Encrypted images
│   ├── decrypted/              # Decrypted images
│   ├── analysis/               # ROI visualizations
│   └── metadata/               # Encryption metadata
└── logs/                       # Execution logs
```

## External Repositories

### Repository A: FlexiMo (IEEE TGRS)
- **URL**: https://github.com/danfenghong/IEEE_TGRS_Fleximo
- **Purpose**: Vision Transformer (DOFA ViT) for semantic segmentation
- **Usage**: Separates ROI (important features) from background

### Repository B: Quantum Image Encryption
- **URL**: https://github.com/ManavMNair/Quantum-image-encryption
- **Purpose**: NEQR quantum encoding + scrambling + DNA encryption
- **Usage**: Encrypts each 8×8 ROI block using quantum circuits

## Zero Data Loss Policy

The system guarantees **pixel-perfect reconstruction**:
- **PSNR**: ∞ dB (perfect match)
- **SSIM**: 1.0000 (perfect structural similarity)
- **Max pixel difference**: 0

## Performance Expectations

| Image Size | ROI Blocks | Estimated Time (shots=1024) |
|------------|-----------|----------------------------|
| 256×256    | ~1,000    | 10-20 minutes              |
| 512×512    | ~4,000    | 30-60 minutes              |
| 1024×1024  | ~16,000   | 2-4 hours                  |

**Note**: Quantum simulation is computationally expensive. Each block requires quantum circuit creation, gate application, and measurement with 1024+ shots on AerSimulator.
