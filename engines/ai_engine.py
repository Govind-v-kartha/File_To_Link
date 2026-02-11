"""
AI Engine - FlexiMo Integration for Semantic Segmentation.

This engine integrates the FlexiMo Vision Transformer (DOFA ViT) from Repository A
for AI-based semantic segmentation of satellite images. It separates important
features (ROI) from non-important features (background).

Repository: https://github.com/danfenghong/IEEE_TGRS_Fleximo
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.logger import setup_logger, get_config_path, load_config

logger = setup_logger("AI_ENGINE", get_config_path())


def _verify_fleximo_repo(repo_path: str) -> bool:
    """
    Verify that the FlexiMo repository exists and has required files.

    Args:
        repo_path: Path to the FlexiMo repository.

    Returns:
        True if repo is valid.

    Raises:
        RuntimeError: If repo is missing or invalid.
    """
    if not os.path.isdir(repo_path):
        raise RuntimeError(
            f"FlexiMo repository NOT FOUND at: {repo_path}\n"
            f"Please clone: git clone https://github.com/danfenghong/IEEE_TGRS_Fleximo {repo_path}"
        )

    # Check for key files
    required_dirs = ["fleximo", "pixel_tasks"]
    for d in required_dirs:
        full_path = os.path.join(repo_path, d)
        if not os.path.isdir(full_path):
            raise RuntimeError(
                f"FlexiMo repository is incomplete: missing '{d}/' directory in {repo_path}"
            )

    required_files = [
        os.path.join("fleximo", "wave_dynamic_layer.py"),
        os.path.join("pixel_tasks", "models_dwv_upernet_256_16.py"),
    ]
    for rf in required_files:
        if not os.path.isfile(os.path.join(repo_path, rf)):
            raise RuntimeError(
                f"FlexiMo repository is incomplete: missing file '{rf}' in {repo_path}"
            )

    logger.info(f"FlexiMo repository verified at: {repo_path}")
    return True


def _download_weights_if_needed(weights_path: str, weights_url: str) -> str:
    """
    Download pretrained weights if they don't exist.

    Args:
        weights_path: Expected path to the weights file.
        weights_url: URL to download from.

    Returns:
        Path to the weights file.
    """
    if os.path.isfile(weights_path):
        logger.info(f"Pretrained weights found: {weights_path}")
        return weights_path

    logger.info(f"Downloading pretrained weights from: {weights_url}")
    logger.info("This may take a while depending on your internet connection...")

    try:
        import urllib.request
        os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
        urllib.request.urlretrieve(weights_url, weights_path)
        logger.info(f"Weights downloaded successfully: {weights_path}")
        return weights_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to download pretrained weights: {e}\n"
            f"Please manually download from: {weights_url}\n"
            f"And place at: {weights_path}"
        )


def _load_fleximo_model(repo_path: str, weights_path: str, device: str = "cpu"):
    """
    Load the FlexiMo ViT segmentation model with pretrained weights.

    Args:
        repo_path: Path to the FlexiMo repository.
        weights_path: Path to the DOFA pretrained weights.
        device: Device to load model on ('cpu' or 'cuda').

    Returns:
        Loaded FlexiMo model ready for inference.
    """
    # Add repo to Python path
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        logger.info(f"Added FlexiMo repo to Python path: {repo_path}")

    # Import the segmentation model from pixel_tasks
    try:
        from pixel_tasks.models_dwv_upernet_256_16 import vit_base_patch16_16, OFAViT
        logger.info("FlexiMo segmentation model (OFAViT UperNet 256x16) imported successfully")
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import FlexiMo model from {repo_path}: {e}\n"
            "Ensure the repository is complete and all dependencies are installed."
        )

    # Create model instance (binary segmentation: 2 classes)
    model = vit_base_patch16_16(num_classes=2)
    logger.info("FlexiMo ViT model instantiated: vit_base_patch16_16(num_classes=2)")

    # Load pretrained DOFA backbone weights
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)

        # Handle pos_embed size mismatch: the checkpoint was saved for img_size=224
        # (196 patches + 1 CLS = 197) but our model uses img_size=256 (256 patches + 1
        # CLS = 257). The model's update_pos_embed() handles runtime resizing, so we
        # skip loading pos_embed and let the model interpolate it on the first forward.
        if "pos_embed" in checkpoint:
            ckpt_pe_shape = checkpoint["pos_embed"].shape
            model_pe_shape = model.pos_embed.shape
            if ckpt_pe_shape != model_pe_shape:
                logger.info(
                    f"pos_embed shape mismatch: checkpoint={list(ckpt_pe_shape)}, "
                    f"model={list(model_pe_shape)}. Interpolating..."
                )
                # Interpolate pos_embed from checkpoint to model size
                ckpt_pe = checkpoint["pos_embed"]  # [1, 197, 768]
                cls_pe = ckpt_pe[:, :1, :]  # [1, 1, 768]
                patch_pe = ckpt_pe[:, 1:, :]  # [1, 196, 768]

                # Reshape patch pos_embed to spatial grid, interpolate, flatten
                old_num = patch_pe.shape[1]
                old_size = int(old_num ** 0.5)  # 14
                new_num = model_pe_shape[1] - 1  # 256
                new_size = int(new_num ** 0.5)  # 16

                patch_pe = patch_pe.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
                patch_pe = F.interpolate(
                    patch_pe, size=(new_size, new_size),
                    mode="bilinear", align_corners=False,
                )
                patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_num, -1)

                checkpoint["pos_embed"] = torch.cat([cls_pe, patch_pe], dim=1)
                logger.info(
                    f"pos_embed interpolated: {list(ckpt_pe_shape)} -> "
                    f"{list(checkpoint['pos_embed'].shape)}"
                )

        msg = model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Pretrained DOFA weights loaded from: {weights_path}")
        logger.info(f"  Missing keys: {len(msg.missing_keys)} (expected - segmentation head)")
        logger.info(f"  Unexpected keys: {len(msg.unexpected_keys)}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load pretrained weights from {weights_path}: {e}"
        )

    model = model.to(device)
    model.eval()
    logger.info(f"FlexiMo model loaded to device: {device}")
    logger.info(f"FlexiMo model loaded from repos/fleximo_repo/")

    return model


def _preprocess_image_for_fleximo(
    image: np.ndarray, target_size: int = 256
) -> torch.Tensor:
    """
    Preprocess an image for FlexiMo inference.

    Resizes to target_size x target_size and normalizes to [0, 1].

    Args:
        image: NumPy array of shape (H, W, 3), dtype uint8.
        target_size: Target spatial size (default 256 for the 256x16 model).

    Returns:
        Torch tensor of shape (1, 3, target_size, target_size).
    """
    # Convert to PIL, resize, back to numpy
    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
    img_resized = np.array(pil_img, dtype=np.float32) / 255.0

    # Convert to torch: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0)

    logger.info(
        f"Image preprocessed for FlexiMo: {image.shape} -> {tensor.shape}, "
        f"range=[{tensor.min():.3f}, {tensor.max():.3f}]"
    )
    return tensor


def segment_image_fleximo(
    image: np.ndarray,
    config: dict = None,
) -> tuple:
    """
    Perform semantic segmentation using FlexiMo ViT backbone features.

    The DOFA ViT backbone is pretrained and produces meaningful feature
    representations.  The UperNet segmentation head is NOT fine-tuned
    (72 missing keys), so we bypass it and compute a **saliency map**
    directly from the backbone's patch-level features:

        saliency(patch) = std(features) * ||features||

    High saliency  = visually complex region (buildings, aircraft, roads) = ROI
    Low  saliency  = uniform / featureless area (desert, empty tarmac)   = BG

    The saliency map is combined with edge-density information from the
    original image and cleaned up with morphological operations.

    Args:
        image: Input satellite image as NumPy array (H, W, 3), dtype uint8.
        config: Configuration dictionary. If None, loads from config.json.

    Returns:
        Tuple of:
            - roi_mask: Binary mask (H, W), uint8, 1 = ROI, 0 = background.
            - background_mask: Binary mask (H, W), uint8, 1 = background, 0 = ROI.
            - saliency_map: Normalised saliency map (H, W) float32.
    """
    if config is None:
        config = load_config()

    # Get paths from config
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_path = os.path.join(project_root, config["repos"]["fleximo"]["path"])
    weights_filename = config["repos"]["fleximo"]["model_weights"]
    weights_url = config["repos"]["fleximo"]["weights_url"]
    weights_path = os.path.join(project_root, weights_filename)

    # Segmentation config
    seg_config = config.get("segmentation", {})
    target_size = seg_config.get("img_size", 256)
    wavelengths = seg_config.get("wavelengths_rgb", [0.665, 0.56, 0.49])

    original_h, original_w = image.shape[:2]

    # ------------------------------------------------------------------
    # Step 1-4: Load model & preprocess (unchanged)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STARTING FlexiMo SEMANTIC SEGMENTATION")
    logger.info("=" * 60)
    _verify_fleximo_repo(repo_path)
    weights_path = _download_weights_if_needed(weights_path, weights_url)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = _load_fleximo_model(repo_path, weights_path, device)
    input_tensor = _preprocess_image_for_fleximo(image, target_size)
    input_tensor = input_tensor.to(device)

    # ------------------------------------------------------------------
    # Step 5: Extract backbone features (PRETRAINED — reliable)
    # ------------------------------------------------------------------
    # We run only the backbone (patch_embed → pos_embed → transformer
    # blocks) and skip the untrained UperNet decoder entirely.
    logger.info(f"Extracting DOFA backbone features (wavelengths={wavelengths})")
    with torch.no_grad():
        wavelist = torch.tensor(wavelengths, device=device).float()
        x, _ = model.patch_embed(input_tensor, wavelist)

        num_patches = x.shape[1]
        if num_patches != (model.pos_embed.shape[1] - 1):
            model.update_pos_embed(num_patches)
        pos_embed = model.pos_embed[:, 1:, :]
        x = x + pos_embed

        for block in model.blocks:
            x = block(x)
        backbone_feat = x  # (1, num_patches, 768)

    patch_grid = int(backbone_feat.shape[1] ** 0.5)  # 16
    feat_spatial = backbone_feat[0].reshape(patch_grid, patch_grid, -1)  # (16,16,768)

    # Saliency = std(features) × ||features||
    # std captures *variety* of activations (complex content), norm
    # captures overall magnitude.
    feat_std = torch.std(feat_spatial, dim=-1).cpu().numpy()   # (16,16)
    feat_norm = torch.norm(feat_spatial, dim=-1).cpu().numpy()  # (16,16)
    saliency = feat_std * feat_norm

    logger.info(
        f"Backbone feature grid: {patch_grid}×{patch_grid}, "
        f"saliency range: [{saliency.min():.2f}, {saliency.max():.2f}]"
    )

    # ------------------------------------------------------------------
    # Step 6: Upsample saliency to original image size
    # ------------------------------------------------------------------
    saliency_up = np.array(
        Image.fromarray(saliency.astype(np.float32)).resize(
            (original_w, original_h), Image.BILINEAR
        )
    )
    s_min, s_max = saliency_up.min(), saliency_up.max()
    saliency_norm = (saliency_up - s_min) / (s_max - s_min + 1e-8)

    # ------------------------------------------------------------------
    # Step 7: Edge-density map from original image (fine-grained detail)
    # ------------------------------------------------------------------
    gray = np.mean(image.astype(np.float32), axis=-1)
    gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    edge_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Smooth to create density (local average of edge magnitudes)
    from scipy.ndimage import uniform_filter
    edge_density = uniform_filter(edge_mag, size=16)
    e_min, e_max = edge_density.min(), edge_density.max()
    edge_norm = (edge_density - e_min) / (e_max - e_min + 1e-8)

    logger.info(
        f"Edge density range: [{edge_density.min():.2f}, {edge_density.max():.2f}]"
    )

    # ------------------------------------------------------------------
    # Step 8: Combine backbone saliency + edge density
    # ------------------------------------------------------------------
    combined = 0.6 * saliency_norm + 0.4 * edge_norm

    # ------------------------------------------------------------------
    # Step 9: Adaptive thresholding (Otsu)
    # ------------------------------------------------------------------
    from skimage.filters import threshold_otsu
    try:
        thresh = threshold_otsu(combined)
    except ValueError:
        thresh = 0.5
    roi_mask = (combined > thresh).astype(np.uint8)

    logger.info(
        f"Otsu threshold: {thresh:.4f}, "
        f"raw ROI pixels: {np.sum(roi_mask)} "
        f"({100 * np.sum(roi_mask) / (original_h * original_w):.1f}%)"
    )

    # ------------------------------------------------------------------
    # Step 10: Morphological cleanup
    # ------------------------------------------------------------------
    from scipy.ndimage import (
        binary_closing, binary_opening, binary_fill_holes, label
    )

    # Close small gaps to get coherent regions
    roi_mask = binary_closing(
        roi_mask, structure=np.ones((9, 9)), iterations=2
    ).astype(np.uint8)

    # Remove small noise blobs
    roi_mask = binary_opening(
        roi_mask, structure=np.ones((5, 5)), iterations=1
    ).astype(np.uint8)

    # Fill holes inside ROI regions
    roi_mask = binary_fill_holes(roi_mask).astype(np.uint8)

    # Remove connected components smaller than min_area pixels
    min_area = 200
    labeled, n_components = label(roi_mask)
    if n_components > 0:
        for comp_id in range(1, n_components + 1):
            comp_mask = labeled == comp_id
            if np.sum(comp_mask) < min_area:
                roi_mask[comp_mask] = 0
        logger.info(
            f"Morphological cleanup: {n_components} components, "
            f"removed those < {min_area} px"
        )

    # ------------------------------------------------------------------
    # Finalise masks
    # ------------------------------------------------------------------
    background_mask = 1 - roi_mask

    total_pixels = original_h * original_w
    roi_count = int(np.sum(roi_mask))

    # Safety: if mask is degenerate, fall back to upper-quartile threshold
    if roi_count == 0 or roi_count == total_pixels:
        logger.warning("Otsu produced degenerate mask; using 75th-percentile threshold")
        thresh = np.percentile(combined, 75)
        roi_mask = (combined > thresh).astype(np.uint8)
        background_mask = 1 - roi_mask
        roi_count = int(np.sum(roi_mask))

    logger.info(
        f"Segmentation complete: {roi_count} ROI pixels, "
        f"{int(np.sum(background_mask))} background pixels"
    )
    logger.info(
        f"ROI coverage: {100 * roi_count / total_pixels:.1f}% of image"
    )

    assert set(np.unique(roi_mask)).issubset({0, 1}), "ROI mask must be binary"
    assert set(np.unique(background_mask)).issubset({0, 1}), "BG mask must be binary"
    assert np.all(roi_mask + background_mask == 1), "Masks must cover full image"

    logger.info("FlexiMo segmentation verified: masks are binary and complementary")
    logger.info("=" * 60)

    # Free model and tensors
    del model, input_tensor, backbone_feat, feat_spatial
    import gc; gc.collect()

    return roi_mask, background_mask, saliency_norm


def save_segmentation_visualization(
    image: np.ndarray,
    roi_mask: np.ndarray,
    background_mask: np.ndarray,
    output_dir: str,
    filename_prefix: str = "segmentation",
) -> list:
    """
    Save visualization of the segmentation results.

    Creates and saves:
    1. Original image with ROI overlay
    2. ROI mask
    3. Background mask
    4. Side-by-side comparison

    Args:
        image: Original image (H, W, 3).
        roi_mask: Binary ROI mask (H, W).
        background_mask: Binary background mask (H, W).
        output_dir: Directory to save visualizations.
        filename_prefix: Prefix for output filenames.

    Returns:
        List of paths to saved visualization files.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # 1. ROI overlay on original
    overlay = image.copy()
    roi_color = np.array([255, 0, 0], dtype=np.uint8)  # Red overlay for ROI
    roi_indices = roi_mask > 0
    overlay[roi_indices] = (
        0.6 * overlay[roi_indices] + 0.4 * roi_color
    ).astype(np.uint8)

    path = os.path.join(output_dir, f"{filename_prefix}_roi_overlay.png")
    Image.fromarray(overlay).save(path)
    saved_files.append(path)

    # 2. ROI extracted
    roi_extracted = image.copy()
    roi_extracted[background_mask > 0] = 0
    path = os.path.join(output_dir, f"{filename_prefix}_roi_extracted.png")
    Image.fromarray(roi_extracted).save(path)
    saved_files.append(path)

    # 3. Background extracted
    bg_extracted = image.copy()
    bg_extracted[roi_mask > 0] = 0
    path = os.path.join(output_dir, f"{filename_prefix}_background_extracted.png")
    Image.fromarray(bg_extracted).save(path)
    saved_files.append(path)

    # 4. Side-by-side comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title("ROI Overlay (Red)", fontsize=14)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(roi_extracted)
    axes[1, 0].set_title(f"ROI Extracted ({np.sum(roi_mask)} pixels)", fontsize=14)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(bg_extracted)
    axes[1, 1].set_title(
        f"Background Extracted ({np.sum(background_mask)} pixels)", fontsize=14
    )
    axes[1, 1].axis("off")

    plt.suptitle("FlexiMo DOFA Backbone — Saliency-Based ROI Segmentation", fontsize=16, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, f"{filename_prefix}_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    saved_files.append(path)

    # 5. Save masks as images
    path = os.path.join(output_dir, f"{filename_prefix}_roi_mask.png")
    Image.fromarray((roi_mask * 255).astype(np.uint8)).save(path)
    saved_files.append(path)

    path = os.path.join(output_dir, f"{filename_prefix}_background_mask.png")
    Image.fromarray((background_mask * 255).astype(np.uint8)).save(path)
    saved_files.append(path)

    logger.info(f"Segmentation visualizations saved to {output_dir}:")
    for f in saved_files:
        logger.info(f"  - {os.path.basename(f)}")

    return saved_files
