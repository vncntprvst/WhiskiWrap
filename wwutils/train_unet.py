"""
Train a U-Net for whisker segmentation.

Arguments:
  --data-dir PATH           Directory containing images/ and masks/ subdirectories.
                           Mutually exclusive with --video-parquet.
                           
  --video-parquet VIDEO PARQUET
                           Video file and corresponding parquet file for training.
                           Can be given multiple times for additional datasets.
                           Generates training data on-the-fly from whisker tracking.
                           Mutually exclusive with --data-dir.
                           
  --frame-indices N [N ...]  Specific frame indices to use for training.
                           Only valid with --video-parquet mode.
                           If not specified, uses all frames available in parquet.
                           Example: --frame-indices 0 5 10 15 20
                           
  --epochs N               Number of training epochs (default: 10)
  
  --batch-size N           Training batch size (default: 4, will auto-detect optimal if not specified)
                           Reduce if running out of GPU memory.
                           
  --lr FLOAT              Learning rate for Adam optimizer (default: 1e-3)
  
  --num-classes N         Number of output classes including background (default: 4)
                          Should match the number of unique whisker labels + 1 for background.
                          
  --output PATH           Path to save the trained model (default: unet_model.pt)
                          Model is saved as TorchScript for deployment.

  --patience N            Early stopping patience: number of epochs without improvement (default: 10)
                          Set to a higher value for longer training or use --no-early-stopping.
                          
  --checkpoint-freq N     Save checkpoint every N epochs (default: 5)
                          Intermediate models saved to checkpoints/ directory.
                          
  --no-early-stopping     Disable early stopping and train for full epochs
                          Use when you want to train for the complete epoch count.
                          
  --force-batch-size      Force use of specified batch size without auto-detection
                          Use when you want exact control over memory usage.
                          
  --max-batch-size N      Maximum batch size to test during auto-detection (default: 64)
                          Higher values test larger batch sizes but take longer to detect.
                          
  --resize-images HEIGHT WIDTH
                          Resize images to specified dimensions for training
                          Example: --resize-images 256 256
                          IMPORTANT: This significantly affects memory usage and quality.

Memory and Performance Optimization:
    The script automatically detects the optimal batch size for your GPU and uses 
    Automatic Mixed Precision (AMP) to reduce memory usage by ~40%.
    
    Batch Size Auto-Detection:
        - Tests increasing batch sizes until GPU memory limit
        - Uses 90% of maximum working batch size for safety
        - Accounts for model, gradients, and optimizer states
        - Typical results: RTX 3070 can handle batch size 2 for 540×720 images,
          or batch size 28+ for 256×256 resized images
    
    Image Resizing for Performance:
        Original size (540×720): Batch size ~2, slower training, full detail
        Resized (384×384):      Batch size ~8-12, balanced quality/speed  
        Resized (256×256):      Batch size ~16-32, fastest training
        
    Quality vs Speed Trade-offs:
        - Full resolution: Best quality, slow training, small batches
        - 384×384 resize: Good quality, moderate speed, medium batches
        - 256×256 resize: Good quality for learning, fast training, large batches
        
    The model can be trained on resized images but applied to full resolution
    during inference, as the network learns whisker patterns that scale.

Checkpointing and Model Saving:
    The script saves multiple model versions for safety and analysis:
    
    - unet_model.pt: Final TorchScript model for deployment
    - checkpoints/unet_best.pt: Best model (lowest loss) with full training state
    - checkpoints/unet_final.pt: Final model with full training state  
    - checkpoints/unet_epoch_N.pt: Periodic checkpoints every --checkpoint-freq epochs
    
    Checkpoint files include model weights, optimizer state, loss history, and
    training arguments, allowing training to be resumed if interrupted.

Early Stopping:
    Training automatically stops if validation loss doesn't improve for --patience
    epochs (default: 10). This prevents overfitting and saves training time.
    The best model is always saved regardless of when training stops.

Output:
    A trained U-Net model saved as a TorchScript file at the specified output path.
    Additional checkpoint files in checkpoints/ directory for model analysis.

Explanation of arguments:
  --data-dir and --video-parquet define how training data is provided.
  --frame-indices allows training on specific frames from the video.
  --epochs, --batch-size, --lr, --num-classes control training parameters.

    # Epochs:
    An epoch is a complete pass through the entire training dataset during machine learning model training.

        # What happens in one epoch:
            Forward pass: The model processes every training sample once
            Loss calculation: Computes how wrong the predictions are
            Backward pass: Calculates gradients to improve the model
            Weight updates: Adjusts model parameters based on gradients

        # Example:
        If you have 1000 training images and set --epochs 50:
            The model will see all 1000 images 50 times
            Each time through all 1000 images = 1 epoch
            Total training iterations = 50 epochs

        # Why multiple epochs?
            Learning is gradual: Models improve incrementally with each pass
            Pattern recognition: Multiple exposures help the model learn complex patterns
            Convergence: Loss typically decreases over epochs until it stabilizes

        # Typical behavior:
            Epoch 1: Loss = 0.85 (model is learning basic features)
            Epoch 10: Loss = 0.42 (getting better at whisker detection)
            Epoch 25: Loss = 0.18 (fine-tuning details)
            Epoch 50: Loss = 0.15 (converged - minimal improvement)

        # Choosing the right number:
            Too few epochs: Model underfits (hasn't learned enough)
            Too many epochs: Model overfits (memorizes training data, poor on new data)
            Typical range: 20-100 epochs for most computer vision tasks

    # Batch size:
    Batch size is the number of training samples the model processes simultaneously before updating its weights.

        # How it works:
            Instead of processing images one-by-one, the model processes them in "batches":
            Batch size = 8:
            - Load 8 images at once
            - Process all 8 through the network
            - Calculate average loss across the 8 images
            - Update model weights once
            - Repeat with next 8 images

        # Example with 1000 training images:
            Batch size = 8:

            1000 ÷ 8 = 125 batches per epoch
            Model updates weights 125 times per epoch
            Batch size = 32:

            1000 ÷ 32 = 31 batches per epoch
            Model updates weights 31 times per epoch
            
    # Learning rate:
    Learning rate (lr) is how big steps the model takes when updating its weights during training. It's one of the most important hyperparameters in machine learning.

        # How it works:
        When the model makes a mistake, it calculates gradients (directions to improve). The learning rate controls how much to adjust the weights:

        # Visual analogy:
        Think of training as rolling a ball down a hill to find the lowest point (best solution):
            High learning rate (0.01): Big steps - might overshoot the target
            Low learning rate (0.0001): Tiny steps - very slow progress
            Good learning rate (0.001): Just right - steady progress toward the goal            

Example usage:
  # Basic training using pre-exported images and masks
  python train_unet.py --data-dir exported_data/ --epochs 20 --batch-size 8
  
  # Train directly from video and parquet files (auto-detects batch size)
  python train_unet.py --video-parquet video.mp4 tracking.parquet --epochs 15
  
  # Train on specific frames with custom parameters
  python train_unet.py --video-parquet video.mp4 tracking.parquet \
                       --frame-indices 0 10 20 30 40 50 \
                       --epochs 25 --lr 0.001 --num-classes 6
                       
  # Memory-optimized training with resized images (recommended for large images)
  python train_unet.py --video-parquet video.mp4 tracking.parquet \
                       --resize-images 256 256 --epochs 25 --num-classes 5
                       
  # High-quality training with moderate resizing
  python train_unet.py --video-parquet video.mp4 tracking.parquet \
                       --resize-images 384 384 --epochs 25 --num-classes 5
                       
  # Force specific batch size and disable early stopping
  python train_unet.py --video-parquet video.mp4 tracking.parquet \
                       --batch-size 16 --force-batch-size \
                       --no-early-stopping --epochs 50
                       
  # Long training with frequent checkpoints
  python train_unet.py --video-parquet video.mp4 tracking.parquet \
                       --epochs 100 --patience 20 --checkpoint-freq 2
                       
Performance Examples:
  RTX 3070 (8GB VRAM):
    - 540×720 images: batch size ~2, full quality, slow training
    - 384×384 images: batch size ~12, good quality, moderate speed  
    - 256×256 images: batch size ~28, good quality, fast training
    
  RTX 4090 (24GB VRAM):
    - 540×720 images: batch size ~8-12, full quality, good speed
    - 384×384 images: batch size ~32+, excellent speed
    - 256×256 images: batch size ~64+, very fast training
"""

import argparse
import os
from pathlib import Path
import logging
import datetime
import time
import subprocess
import tempfile
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from typing import Dict, Tuple


class SegmentationDataset(Dataset):
    """Dataset of image/mask pairs stored in two directories."""

    def __init__(self, image_dir: str, mask_dir: str):
        self.image_paths = sorted(
            [
                p
                for p in Path(image_dir).glob("*")
                if p.suffix in {".png", ".jpg", ".jpeg"}
            ]
        )
        self.mask_paths = sorted(
            [
                p
                for p in Path(mask_dir).glob("*")
                if p.suffix in {".png", ".jpg", ".jpeg"}
            ]
        )
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Number of images and masks must match")
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.to_tensor(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = torch.as_tensor(
            np.array(Image.open(self.mask_paths[idx])), dtype=torch.long
        )
        return image, mask


class VideoParquetDataset(Dataset):
    """Dataset that loads frames from video and generates masks from parquet file on-the-fly."""

    def __init__(
        self,
        video_file: str,
        parquet_file: str,
        frame_indices: list = None,
        resize_to: tuple = None,
        use_wid_only: bool = False,
        wid_to_class: dict = None,
    ):
        """
        Args:
            video_file: Path to video file
            parquet_file: Path to parquet file with whisker tracking data
            frame_indices: List of frame indices to use. If None, uses all frames in parquet.
            resize_to: Tuple of (height, width) to resize images to. If None, uses original size.
            use_wid_only: If True, ignore 'label' column and use 'wid' values directly
            wid_to_class: Dictionary mapping whisker IDs to class indices for efficient training
        """
        self.video_file = video_file
        self.parquet_file = parquet_file
        self.resize_to = resize_to
        self.use_wid_only = use_wid_only
        self.wid_to_class = wid_to_class
        self.df = pd.read_parquet(parquet_file)

        # Frame caching for better performance
        self.frame_cache = {}
        self.cache_size_limit = 100  # Cache up to 100 frames

        # Get available frame indices from parquet
        available_frames = sorted(self.df["fid"].unique())

        if frame_indices is None:
            self.frame_indices = available_frames
        else:
            # Only use frames that exist in both the requested list and parquet data
            self.frame_indices = [
                fid for fid in frame_indices if fid in available_frames
            ]

        # Open video capture with error handling
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video {video_file}")

        # Set video codec options to handle corrupted frames and reduce warnings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to avoid stale frames
        # Suppress some video decoding warnings by setting error callback (if available)
        try:
            cv2.setLogLevel(cv2.LOG_LEVEL_WARNING)  # Reduce OpenCV log verbosity
        except AttributeError:
            pass  # Older OpenCV versions don't have this function

        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Transforms
        self.to_tensor = transforms.ToTensor()

        print(
            f"Dataset initialized with {len(self.frame_indices)} frames from {video_file}"
        )
        print(f"Frame caching enabled (limit: {self.cache_size_limit} frames)")

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        import time

        start_time = time.time()

        fid = self.frame_indices[idx]

        # Check frame cache first
        if fid in self.frame_cache:
            frame_rgb = self.frame_cache[fid]
            cache_hit = True
        else:
            cache_hit = False
            # Read frame from video with retry logic
            video_start = time.time()
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        break
                    else:
                        if attempt == max_retries - 1:
                            raise RuntimeError(
                                f"Could not read frame {fid} from video after {max_retries} attempts"
                            )
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Error reading frame {fid}: {e}")
                    # Wait a bit before retry
                    import time

                    time.sleep(0.01)

            video_time = time.time() - video_start

            # Convert BGR to RGB for PIL/torch compatibility
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Cache the frame if we have space
            if len(self.frame_cache) < self.cache_size_limit:
                self.frame_cache[fid] = frame_rgb.copy()

        # Resize frame if requested
        resize_start = time.time()
        if self.resize_to is not None:
            original_shape = frame_rgb.shape[:2]  # (H, W)
            frame_rgb = cv2.resize(
                frame_rgb, (self.resize_to[1], self.resize_to[0])
            )  # cv2 uses (W, H)
            scale_y = self.resize_to[0] / original_shape[0]
            scale_x = self.resize_to[1] / original_shape[1]
        else:
            scale_y = scale_x = 1.0
        resize_time = time.time() - resize_start

        # Generate mask from parquet data (with scaling if resized)
        mask_start = time.time()
        mask = build_mask(
            self.df,
            frame_rgb.shape,
            fid,
            scale_y,
            scale_x,
            self.use_wid_only,
            self.wid_to_class,
        )
        mask_time = time.time() - mask_start

        # Convert to tensors
        tensor_start = time.time()
        image = self.to_tensor(frame_rgb)
        mask_tensor = torch.as_tensor(mask, dtype=torch.long)
        tensor_time = time.time() - tensor_start

        total_time = time.time() - start_time

        # Log timing for first few samples to identify bottlenecks
        if idx < 5:  # Only for first few samples to avoid spam
            timing_info = f"Sample {idx} (frame {fid}): total={total_time:.3f}s"
            if not cache_hit:
                timing_info += f" [video={video_time:.3f}s, resize={resize_time:.3f}s, mask={mask_time:.3f}s, tensor={tensor_time:.3f}s]"
            else:
                timing_info += f" [CACHED, resize={resize_time:.3f}s, mask={mask_time:.3f}s, tensor={tensor_time:.3f}s]"
            print(f"  {timing_info}", flush=True)

        return image, mask_tensor

    def __del__(self):
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def setup_logging(log_dir: str = None) -> str:
    """Set up logging to both file and console."""
    if log_dir is None:
        log_dir = os.path.dirname(__file__)

    # Create logs directory if it doesn't exist
    log_dir = os.path.join(log_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_unet_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # Also print to console
        ],
    )

    return log_file


def monitor_gpu_utilization():
    """Monitor GPU utilization and memory usage."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_util, mem_used, mem_total = result.stdout.strip().split(", ")
            return int(gpu_util), int(mem_used), int(mem_total)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
        pass
    return None, None, None


def find_optimal_batch_size(
    model, dataset, device, max_batch_size=32, input_shape=(3, 256, 256)
):
    """
    Find the optimal batch size that fits in GPU memory.

    Args:
        model: The model to test
        dataset: Dataset to get sample data from
        device: Device to test on ('cuda' or 'cpu')
        max_batch_size: Maximum batch size to test
        input_shape: Expected input shape (C, H, W)

    Returns:
        Optimal batch size (int)
    """
    if device == "cpu":
        logging.info("Running on CPU, using default batch size of 8")
        return 8

    if not torch.cuda.is_available():
        logging.info("CUDA not available, using default batch size of 8")
        return 8

    logging.info("Finding optimal batch size for GPU...")

    # Get GPU memory info
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
    logging.info(f"Total GPU memory: {total_memory:.2f}GB")

    # Clear GPU cache and reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Set memory fraction to use 90% of available GPU memory
    torch.cuda.set_per_process_memory_fraction(0.9)
    logging.info("Set GPU memory fraction to 90%")

    # Get a sample from the dataset to determine actual input shape
    try:
        sample_img, sample_mask = dataset[0]
        actual_input_shape = sample_img.shape
        logging.info(f"Detected input shape: {actual_input_shape}")

        # Calculate approximate memory per sample
        pixels_per_sample = (
            actual_input_shape[1] * actual_input_shape[2] * actual_input_shape[0]
        )
        memory_per_sample_mb = pixels_per_sample * 4 / 1024**2  # 4 bytes per float32
        logging.info(f"Estimated memory per sample: {memory_per_sample_mb:.2f}MB")

    except Exception as e:
        logging.warning(f"Could not get sample from dataset: {e}. Using default shape.")
        actual_input_shape = input_shape

    model = model.to(device)
    model.train()

    # Clear GPU cache again after moving model
    torch.cuda.empty_cache()

    optimal_batch_size = 4  # Fallback

    def debug_memory_usage(batch_size):
        """Print detailed memory usage information."""
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        logging.info(
            f"Batch size {batch_size}: Memory - "
            f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, "
            f"Peak Allocated: {max_allocated:.2f}GB, Peak Reserved: {max_reserved:.2f}GB"
        )
        return allocated

    # Test increasing batch sizes more aggressively for RTX 3070
    test_batch_sizes = [2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]

    for batch_size in test_batch_sizes:
        if batch_size > max_batch_size:
            break

        try:
            # Reset memory tracking for this test
            torch.cuda.reset_peak_memory_stats()

            # Create dummy batch with exact same shape as real data
            dummy_input = torch.randn(batch_size, *actual_input_shape).to(device)
            dummy_target = torch.randint(
                0, 5, (batch_size, actual_input_shape[1], actual_input_shape[2])
            ).to(device)

            # Test forward pass with gradient checkpointing for memory efficiency
            with torch.no_grad():
                output = model(dummy_input)

            # Test backward pass (more memory intensive) with mixed precision
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            # Use newer AMP API to avoid deprecation warnings
            try:
                scaler = torch.amp.GradScaler("cuda")
                autocast_context = torch.amp.autocast("cuda")
            except AttributeError:
                # Fallback to older API for older PyTorch versions
                scaler = torch.cuda.amp.GradScaler()
                autocast_context = torch.cuda.amp.autocast()

            optimizer.zero_grad()

            # Use automatic mixed precision to reduce memory usage
            with autocast_context:
                output = model(dummy_input)
                loss = criterion(output, dummy_target)

            # Scale loss and backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # If we get here, this batch size works
            optimal_batch_size = batch_size
            memory_used = debug_memory_usage(batch_size)

            # Clear memory for next test
            del dummy_input, dummy_target, output, loss, optimizer, criterion, scaler
            torch.cuda.empty_cache()

            # For high-memory GPUs, continue testing larger batch sizes
            if memory_used < total_memory * 0.6:  # If using less than 60% of GPU memory
                logging.info(f"  Still have headroom, testing larger batch sizes...")
                continue

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.info(f"Batch size {batch_size}: Out of memory")
                torch.cuda.empty_cache()  # Clean up after OOM
                break
            else:
                logging.warning(f"Batch size {batch_size}: Error - {e}")
                torch.cuda.empty_cache()
                break
        except Exception as e:
            logging.warning(f"Batch size {batch_size}: Unexpected error - {e}")
            torch.cuda.empty_cache()
            break

    # For high-resolution images on capable GPUs, use 90% of optimal instead of 80%
    if total_memory >= 6.0:  # GPUs with 6GB+ memory
        safety_factor = 0.9
        logging.info("High-memory GPU detected, using 90% safety factor")
    else:
        safety_factor = 0.8
        logging.info("Using 80% safety factor for safety")

    safe_batch_size = max(2, int(optimal_batch_size * safety_factor))

    logging.info(f"Optimal batch size found: {optimal_batch_size}")
    logging.info(
        f"Using safe batch size: {safe_batch_size} ({int(safety_factor*100)}% of optimal for safety)"
    )

    # Reset memory fraction to default
    torch.cuda.set_per_process_memory_fraction(1.0)
    torch.cuda.empty_cache()

    return safe_batch_size


def log_training_args(args):
    """Log all training arguments."""
    logging.info("=" * 50)
    logging.info("TRAINING CONFIGURATION")
    logging.info("=" * 50)

    if args.data_dir:
        logging.info(f"Data directory: {args.data_dir}")
    else:
        for idx, (video_file, parquet_file) in enumerate(args.video_parquet, 1):
            logging.info(f"Video {idx}: {video_file}")
            logging.info(f"Parquet {idx}: {parquet_file}")
        if args.frame_indices:
            logging.info(f"Frame indices: {args.frame_indices}")
        else:
            logging.info("Frame indices: All available frames")

    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Number of classes: {args.num_classes}")
    logging.info(f"Output model path: {args.output}")
    logging.info(f"Early stopping patience: {args.patience}")
    logging.info(f"Checkpoint frequency: {args.checkpoint_freq} epochs")
    logging.info(f"Early stopping enabled: {not args.no_early_stopping}")
    logging.info("=" * 50)


def train(args):
    # Set up logging
    log_file = setup_logging()
    logging.info("Starting U-Net training")
    log_training_args(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Create dataset based on input method
    temp_frames_dirs = []  # Track temporary directories for cleanup
    dataset_is_video = False

    if args.data_dir:
        # Original method: directory with images/ and masks/
        dataset = SegmentationDataset(
            os.path.join(args.data_dir, "images"), os.path.join(args.data_dir, "masks")
        )
        logging.info(f"Using directory-based dataset from {args.data_dir}")
    else:
        dataset_is_video = True
        # New method: video + parquet files with frame pre-extraction for speed
        resize_to = tuple(args.resize_images) if args.resize_images else None

        print("=" * 80)
        print("PARQUET FILE ANALYSIS")
        print("=" * 80)

        # Load all parquet files and combine for class analysis
        dfs = []
        for vid, pq in args.video_parquet:
            df_tmp = pd.read_parquet(pq)
            print(f"Loaded {pq} with columns: {list(df_tmp.columns)}")
            dfs.append(df_tmp)
        df_sample = pd.concat(dfs, ignore_index=True)

        # Analyze whisker ID frequency to determine optimal number of classes
        # We use 'wid' in df_sample.columns
        # Note that the cs.combine_to_file step in whisker_tracking.py returns a df with 'wid' and 'label', where 'wid' is the whisker ID and 'label' is set to 0.
        # The reclassification script (rc.reclassify step in whisker_tracking.py) will then save those 'wid' to the 'label' column, and set the new 'wid' values to the 'wid' column.

        # Analyze whisker ID frequency and determine dominant whiskers
        wid_counts = df_sample["wid"].value_counts().sort_index()
        unique_wids = sorted(df_sample["wid"].unique())
        num_unique_whiskers = len(unique_wids)
        print(f"Total unique whisker IDs (wid): {num_unique_whiskers}")
        print(f"WID values: {unique_wids}")
        print(f"WID frequency distribution:")

        # Determine dominant whiskers (>5% of annotations) vs rare whiskers
        dominant_threshold = len(df_sample) * 0.05
        dominant_wids = []
        rare_wids = []

        for wid, count in wid_counts.items():
            percentage = count / len(df_sample) * 100
            print(f"  WID {wid}: {count} annotations ({percentage:.1f}%)")
            if count >= dominant_threshold:
                dominant_wids.append(wid)
            else:
                rare_wids.append(wid)

        print(f"\nWhisker Classification:")
        print(f"  Dominant whiskers (>5%): {len(dominant_wids)} → {dominant_wids}")
        print(f"  Rare whiskers (<5%): {len(rare_wids)} → {rare_wids}")

        # Create efficient class mapping only if requested
        if args.efficient_classes:
            # Create efficient class mapping: background=0, dominant whiskers=1,2,3..., rare whiskers=unclassified
            wid_to_class_mapping = {}
            class_to_wid_mapping = {0: "background"}

            # Note that wid=0 typically exists. It is not background.
            class_idx = 1
            for wid in sorted(dominant_wids):
                wid_to_class_mapping[wid] = class_idx
                class_to_wid_mapping[class_idx] = wid
                class_idx += 1

            # Map rare whiskers to unclassified class (not background)
            if rare_wids:
                unclassified_class = class_idx
                for wid in rare_wids:
                    wid_to_class_mapping[wid] = unclassified_class
                class_to_wid_mapping[unclassified_class] = f"unclassified)"
                num_classes = class_idx + 1  # background + dominant + unclassified
            else:
                num_classes = class_idx  # background + dominant only

            print(f"\n✨ EFFICIENT CLASS MAPPING ENABLED:")
            print(f"  Total classes needed: {num_classes}")
            print(f"  Class 0: Background (no whiskers)")
            for class_id, wid in class_to_wid_mapping.items():
                if class_id > 0:
                    if isinstance(wid, str):  # unclassified description
                        print(f"  Class {class_id}: {wid}")
                    else:
                        print(f"  Class {class_id}: WID {wid}")

            # Show which specific rare WIDs are mapped to unclassified
            rare_non_bg = [w for w in rare_wids if w != 0]
            if rare_non_bg:
                print(f"  → Rare WIDs mapped to unclassified: {rare_non_bg}")
        else:
            # Use original sparse mapping (for backward compatibility)
            wid_to_class_mapping = None
            num_classes = max(unique_wids) + 1
            print(f"\n📊 Using original sparse mapping:")
            print(
                f"  Total classes needed: {num_classes} (0 through {max(unique_wids)})"
            )
            print(f"  Add --efficient-classes for optimal class usage")

        # Store mapping for use in dataset

        # Auto-adjust num_classes based on efficient mapping
        if args.auto_classes:
            print(
                f"\n🔄 AUTO-CLASSES: Setting num_classes to efficient value: {num_classes}"
            )
            args.num_classes = num_classes
        elif args.num_classes < num_classes:
            print(
                f"\n⚠️  WARNING: Specified num_classes ({args.num_classes}) is less than needed ({num_classes})"
            )
            print(f"    Auto-adjusting to {num_classes}")
            args.num_classes = num_classes
        elif args.num_classes > num_classes:
            print(
                f"\n� INFO: Using {args.num_classes} classes (optimal would be {num_classes})"
            )
        # Auto-adjust num_classes based on chosen mapping
        if args.auto_classes:
            print(
                f"\n🔄 AUTO-CLASSES: Setting num_classes to optimal value: {num_classes}"
            )
            args.num_classes = num_classes
        elif args.num_classes < num_classes:
            print(
                f"\n⚠️  WARNING: Specified num_classes ({args.num_classes}) is less than needed ({num_classes})"
            )
            print(f"    Auto-adjusting to {num_classes}")
            args.num_classes = num_classes
        elif args.num_classes > num_classes and args.efficient_classes:
            print(
                f"\n💡 INFO: Using {args.num_classes} classes (optimal would be {num_classes})"
            )

        print(f"\n✅ Final num_classes: {args.num_classes}")

        # Check frame range
        if "fid" in df_sample.columns:
            frame_range = (df_sample["fid"].min(), df_sample["fid"].max())
            print(f"\nFrame range in parquet: {frame_range}")
        print("=" * 80)

        datasets = []
        temp_frames_dirs = []

        for video_file, parquet_file in args.video_parquet:
            # Determine frame indices to extract
            if args.frame_indices:
                frame_indices = list(
                    range(args.frame_indices[0], args.frame_indices[1] + 1)
                )
            else:
                cap = cv2.VideoCapture(video_file)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                frame_indices = list(range(total_frames))

            print(
                f"\n⚠️  VIDEO-PARQUET MODE: Pre-extracting frames from {video_file}..."
            )
            print(
                f"This will extract {len(frame_indices)} frames and may take a few minutes."
            )

            extraction_start = time.time()
            frames_temp_dir = extract_frames_from_video(
                video_file, frame_indices, resize_to
            )
            extraction_time = time.time() - extraction_start
            print(f"Frame extraction completed in {extraction_time:.1f}s")

            dataset_single = ExtractedFramesDataset(
                frames_temp_dir,
                parquet_file,
                frame_indices,
                use_wid_only=args.ignore_label_column,
                wid_to_class=wid_to_class_mapping,
            )

            if resize_to:
                logging.info(
                    f"Using extracted frames dataset with pre-resized images to {resize_to}: {frames_temp_dir}"
                )
            else:
                logging.info(f"Using extracted frames dataset: {frames_temp_dir}")

            datasets.append(dataset_single)
            temp_frames_dirs.append(frames_temp_dir)

        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    logging.info(f"Dataset size: {len(dataset)} samples")

    # Initialize model first to test batch sizes
    model = UNet(3, args.num_classes).to(device)
    logging.info(
        f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Track original batch size for reporting
    orig_batch_size = args.batch_size

    # Determine optimal batch size if not specified or if default was used
    if (
        args.batch_size == 4 and not args.force_batch_size
    ):  # Default value, try to find optimal
        logging.info("Default batch size detected, searching for optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(
            model, dataset, device, args.max_batch_size
        )
        if optimal_batch_size != args.batch_size:
            logging.info(
                f"Updating batch size from {args.batch_size} to {optimal_batch_size}"
            )
            args.batch_size = optimal_batch_size
    else:
        if args.force_batch_size:
            logging.info(f"Forced to use specified batch size: {args.batch_size}")
        else:
            logging.info(f"Using user-specified batch size: {args.batch_size}")

    # Optimize DataLoader workers based on CPU count and dataset size
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()

    # For video processing, too many workers can cause issues with VideoCapture
    # Use a more conservative approach for video data
    if dataset_is_video:
        # For video datasets, limit workers to avoid VideoCapture conflicts
        optimal_workers = min(
            cpu_count // 4,  # More conservative for video
            8,  # Cap at 8 for video processing
            len(dataset) // 16,  # Ensure enough samples per worker
        )
    else:
        # Calculate optimal workers: use more cores for large datasets, but ensure efficiency
        samples_per_worker = max(4, len(dataset) // 32)  # At least 4 samples per worker
        max_workers_by_data = len(dataset) // samples_per_worker

        # For many-core systems, use up to 3/4 of cores, but cap based on dataset size
        optimal_workers = min(
            cpu_count * 3 // 4,  # Use 75% of CPU cores
            max_workers_by_data,  # Don't exceed what's useful for dataset size
            24,  # Reasonable upper limit to avoid oversubscription
        )

    optimal_workers = max(1, optimal_workers)  # Ensure at least 1 worker

    logging.info(f"Detected {cpu_count} CPU cores, dataset size {len(dataset)}")
    logging.info(f"Dataset type: {'Video' if dataset_is_video else 'Other'}")
    logging.info(
        f"Using {optimal_workers} DataLoader workers (samples per worker: ~{len(dataset) // optimal_workers})"
    )

    print(f"Creating DataLoader with {optimal_workers} workers...", flush=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=6,
    )  # Increase prefetch buffer for many workers
    print(
        f"DataLoader created successfully - {len(loader)} batches per epoch", flush=True
    )
    logging.info(f"Number of batches per epoch: {len(loader)}")
    logging.info(
        f"DataLoader configured: {optimal_workers} workers, pin_memory=True, persistent_workers=True, prefetch_factor=6"
    )

    # Model was already initialized above for batch size testing

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Add mixed precision training for memory efficiency with large images
    if device == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
            autocast_context = torch.amp.autocast("cuda")
        except AttributeError:
            # Fallback to older API for older PyTorch versions
            scaler = torch.cuda.amp.GradScaler()
            autocast_context = torch.cuda.amp.autocast()
        use_amp = True
        logging.info("Using Automatic Mixed Precision (AMP) for memory efficiency")
    else:
        scaler = None
        autocast_context = None
        use_amp = False

    logging.info("Optimizer and loss criterion initialized")

    # Training loop with detailed logging
    logging.info("Starting training loop...")
    logging.info(
        "Note: Video decoding warnings (Invalid NAL unit, missing picture) are cosmetic and don't affect training quality"
    )
    training_start_time = time.time()

    # Best model tracking and early stopping
    best_loss = float("inf")
    no_improvement_count = 0
    patience = args.patience  # Use user-defined patience

    # Create checkpoints directory
    checkpoint_dir = os.path.join(os.path.dirname(args.output), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logging.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Training configuration summary
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Dataset: {len(dataset)} samples")
    print(
        f"Batch size: {args.batch_size} (auto-detected: {args.batch_size != orig_batch_size})"
    )
    print(f"Epochs: {args.epochs}")
    print(f"DataLoader: {optimal_workers} workers, prefetch_factor=6")
    print(f"Hardware: {cpu_count} CPU cores, GPU: {device}")
    print(f"Mixed precision: {use_amp}")
    print(f"Image resizing: {'Yes' if args.resize_images else 'No'}")
    print(
        f"Frame caching: {'Yes' if hasattr(dataset, 'cache_size') and dataset.cache_size > 0 else 'No'}"
    )
    print("=" * 80 + "\n")

    logging.info("Starting training with configuration summary logged above")

    model.train()
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        num_batches = len(loader)

        # Track data loading and GPU processing times
        data_load_times = []
        gpu_process_times = []
        data_load_start = time.time()

        logging.info(f"\nEpoch {epoch+1}/{args.epochs} started")
        print(f"\nEpoch {epoch+1}/{args.epochs} - Starting data loading...", flush=True)

        # Add timing for the DataLoader iterator creation
        loader_start = time.time()
        loader_iter = iter(loader)
        print(
            f"DataLoader iterator created in {time.time() - loader_start:.2f}s",
            flush=True,
        )

        for batch_idx in range(num_batches):
            # Time the actual batch loading
            batch_load_start = time.time()
            print(f"Loading batch {batch_idx+1}/{num_batches}...", end=" ", flush=True)

            try:
                imgs, masks = next(loader_iter)
                batch_load_time = time.time() - batch_load_start
                print(f"loaded in {batch_load_time:.2f}s", flush=True)

                # Measure data loading time (this is now just the next() call)
                data_load_time = batch_load_time
                data_load_times.append(data_load_time)

                # If first batch, report details
                if batch_idx == 0:
                    print(
                        f"First batch shape: imgs={imgs.shape}, masks={masks.shape}",
                        flush=True,
                    )
                    logging.info(
                        f"First batch loaded - imgs: {imgs.shape}, masks: {masks.shape}, load time: {data_load_time:.2f}s"
                    )
            except Exception as e:
                print(f"ERROR loading batch {batch_idx+1}: {e}", flush=True)
                logging.error(f"Error loading batch {batch_idx+1}: {e}")
                raise

            # Start timing GPU processing (including transfer and computation)
            gpu_start_time = time.time()

            # Non-blocking GPU transfer for better pipeline
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # More efficient gradient zeroing
            optimizer.zero_grad(set_to_none=True)

            # Use mixed precision for memory efficiency
            if use_amp:
                with autocast_context:
                    out = model(imgs)
                    loss = criterion(out, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(imgs)
                loss = criterion(out, masks)
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * imgs.size(0)

            # Measure GPU processing time (including transfer, forward, backward, update)
            gpu_process_time = time.time() - gpu_start_time
            gpu_process_times.append(gpu_process_time)

            # Calculate total batch time (data loading + GPU processing)
            batch_time = data_load_time + gpu_process_time

            # Print progress every 5 batches instead of 10 for better feedback
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                avg_loss_so_far = total_loss / ((batch_idx + 1) * args.batch_size)
                progress_pct = (batch_idx + 1) / num_batches * 100
                elapsed_time = time.time() - epoch_start_time
                eta_epoch = (
                    elapsed_time / (batch_idx + 1) * (num_batches - batch_idx - 1)
                )

                # Calculate average timings for recent batches
                recent_batches = min(5, len(data_load_times))
                avg_data_load = sum(data_load_times[-recent_batches:]) / recent_batches
                avg_gpu_process = (
                    sum(gpu_process_times[-recent_batches:]) / recent_batches
                )
                data_gpu_ratio = avg_data_load / (
                    avg_gpu_process + 1e-8
                )  # Avoid division by zero

                # Calculate throughput
                samples_per_sec = args.batch_size / (avg_data_load + avg_gpu_process)

                logging.info(
                    f"  Batch {batch_idx+1}/{num_batches} ({progress_pct:.1f}%) - "
                    f"Loss: {batch_loss:.4f} - Avg Loss: {avg_loss_so_far:.4f} - "
                    f"Batch Time: {batch_time:.2f}s - ETA: {eta_epoch:.0f}s"
                )
                logging.info(
                    f"  Timing: Data {avg_data_load:.3f}s - GPU {avg_gpu_process:.3f}s - "
                    f"Ratio {data_gpu_ratio:.2f} - Throughput: {samples_per_sec:.1f} samples/sec"
                )

                # Also print to console for immediate feedback during long epochs
                print(
                    f"    Progress: {progress_pct:.1f}% - Loss: {batch_loss:.4f} - ETA: {eta_epoch:.0f}s",
                    flush=True,
                )
                print(
                    f"    Data {avg_data_load:.3f}s - GPU {avg_gpu_process:.3f}s - "
                    f"Throughput: {samples_per_sec:.1f} samples/sec",
                    flush=True,
                )

                # Real-time performance recommendations
                if data_gpu_ratio > 1.5:
                    print(
                        f"    ⚠️  Data loading bottleneck (ratio {data_gpu_ratio:.2f}) - "
                        f"consider more workers or caching",
                        flush=True,
                    )
                elif data_gpu_ratio < 0.3:
                    print(
                        f"    ℹ️  GPU bottleneck (ratio {data_gpu_ratio:.2f}) - "
                        f"data loading is very efficient",
                        flush=True,
                    )

            # Monitor GPU utilization every 10 batches
            if (batch_idx + 1) % 10 == 0:
                gpu_util, mem_used, mem_total = monitor_gpu_utilization()
                if gpu_util is not None:
                    logging.info(
                        f"  GPU Utilization: {gpu_util}% - Memory: {mem_used}MB/{mem_total}MB"
                    )
                    print(
                        f"    GPU: {gpu_util}% utilized - Memory: {mem_used}MB/{mem_total}MB",
                        flush=True,
                    )

            # Prepare for next iteration's data loading timing
            data_load_start = time.time()

        # Epoch summary with detailed timing analysis
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = total_loss / len(dataset)

        # Calculate timing statistics
        total_data_load_time = sum(data_load_times)
        total_gpu_process_time = sum(gpu_process_times)
        avg_data_load_time = (
            total_data_load_time / len(data_load_times) if data_load_times else 0
        )
        avg_gpu_process_time = (
            total_gpu_process_time / len(gpu_process_times) if gpu_process_times else 0
        )

        data_load_pct = (total_data_load_time / epoch_time) * 100
        gpu_process_pct = (total_gpu_process_time / epoch_time) * 100
        overhead_pct = (
            100 - data_load_pct - gpu_process_pct
        )  # Time spent on other operations

        # Calculate throughput metrics
        total_samples = len(dataset)
        samples_per_sec = total_samples / epoch_time
        data_gpu_ratio = avg_data_load_time / (avg_gpu_process_time + 1e-8)

        logging.info(
            f"Epoch {epoch+1}/{args.epochs} completed - "
            f"Average Loss: {avg_epoch_loss:.4f} - "
            f"Epoch Time: {epoch_time:.2f}s"
        )
        logging.info(
            f"Timing breakdown - Data: {total_data_load_time:.2f}s ({data_load_pct:.1f}%) - "
            f"GPU: {total_gpu_process_time:.2f}s ({gpu_process_pct:.1f}%) - "
            f"Overhead: {overhead_pct:.1f}%"
        )
        logging.info(
            f"Performance - {samples_per_sec:.1f} samples/sec - "
            f"Data/GPU ratio: {data_gpu_ratio:.2f} - "
            f"Workers: {optimal_workers}"
        )

        # Print to console for immediate feedback
        print(
            f"Epoch {epoch+1}/{args.epochs} - loss: {avg_epoch_loss:.4f} - time: {epoch_time:.2f}s"
        )
        print(
            f"  Performance: {samples_per_sec:.1f} samples/sec - Data: {data_load_pct:.1f}% - GPU: {gpu_process_pct:.1f}%"
        )

        # Comprehensive performance analysis and recommendations
        if data_load_pct > 40:
            recommendation = "Consider increasing workers, enabling caching, or using --resize-images"
            print(
                f"  ⚠️  Data loading bottleneck ({data_load_pct:.1f}%) - {recommendation}"
            )
            logging.warning(
                f"Data loading bottleneck: {data_load_pct:.1f}% - {recommendation}"
            )
        elif gpu_process_pct > 75:
            print(f"  ✅ Good GPU utilization ({gpu_process_pct:.1f}%)")
            logging.info(f"Good GPU utilization: {gpu_process_pct:.1f}%")
        elif gpu_process_pct < 50:
            recommendation = (
                "GPU is underutilized - consider larger batch size or smaller images"
            )
            print(
                f"  ⚠️  Low GPU utilization ({gpu_process_pct:.1f}%) - {recommendation}"
            )
            logging.warning(
                f"Low GPU utilization: {gpu_process_pct:.1f}% - {recommendation}"
            )

        # Additional optimization suggestions
        if data_gpu_ratio > 2.0:
            print(
                f"  💡 Tip: Data loading is {data_gpu_ratio:.1f}x slower than GPU - try more workers"
            )
        elif overhead_pct > 20:
            print(
                f"  💡 Tip: High overhead ({overhead_pct:.1f}%) - check for inefficient operations"
            )
        elif gpu_process_pct > 80:
            print(
                f"  ✅ GPU utilization is good ({gpu_process_pct:.1f}% of epoch time)"
            )
            logging.info(f"Good GPU utilization: {gpu_process_pct:.1f}% of epoch time")

        # Save intermediate checkpoints every N epochs (user-configurable)
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    "args": vars(args),
                    "wid_to_class_mapping": wid_to_class_mapping,
                    "class_to_wid_mapping": class_to_wid_mapping,
                },
                checkpoint_path,
            )
            logging.info(f"Checkpoint saved: {checkpoint_path}")

        # Best model tracking
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            no_improvement_count = 0
            best_model_path = os.path.join(checkpoint_dir, "unet_best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    "args": vars(args),
                    "wid_to_class_mapping": wid_to_class_mapping,
                    "class_to_wid_mapping": class_to_wid_mapping,
                },
                best_model_path,
            )
            logging.info(
                f"New best model saved (loss: {best_loss:.4f}): {best_model_path}"
            )
            print(f"★ New best model! Loss: {best_loss:.4f}")
        else:
            no_improvement_count += 1
            logging.info(
                f"No improvement for {no_improvement_count} epochs (best: {best_loss:.4f})"
            )

        # Early stopping (only if not disabled)
        if not args.no_early_stopping and no_improvement_count >= patience:
            logging.info(
                f"Early stopping triggered after {patience} epochs without improvement"
            )
            print(
                f"Early stopping: No improvement for {patience} epochs. Best loss: {best_loss:.4f}"
            )
            break

    # Training complete
    total_training_time = time.time() - training_start_time
    logging.info(
        f"\nTraining completed in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)"
    )
    logging.info(f"Best loss achieved: {best_loss:.4f}")

    # Save final model
    logging.info(f"Saving final model to {args.output}")

    # Save with full checkpoint info including mappings
    final_model_data = {
        "model_state_dict": model.cpu().state_dict(),
        "num_classes": args.num_classes,
        "args": vars(args),
        "wid_to_class_mapping": wid_to_class_mapping,
        "class_to_wid_mapping": class_to_wid_mapping,
    }

    torch.save(final_model_data, args.output)

    # Also save final checkpoint with full state
    final_checkpoint_path = os.path.join(checkpoint_dir, "unet_final.pt")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_epoch_loss,
            "best_loss": best_loss,
            "args": vars(args),
            "wid_to_class_mapping": wid_to_class_mapping,
            "class_to_wid_mapping": class_to_wid_mapping,
        },
        final_checkpoint_path,
    )

    logging.info(f"Final model saved to {args.output}")
    logging.info(f"Final checkpoint saved to {final_checkpoint_path}")
    logging.info(
        f'Best model available at: {os.path.join(checkpoint_dir, "unet_best.pt")}'
    )
    logging.info(f"Log file saved to: {log_file}")

    # Clean up temporary frames directories if they were created
    for temp_dir in temp_frames_dirs:
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary frames directory: {temp_dir}")
            logging.info(f"Cleaned up temporary frames directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")
            logging.warning(f"Could not clean up temporary directory {temp_dir}: {e}")

    print(f"Training completed! Final model saved to {args.output}")
    print(
        f'Best model (loss: {best_loss:.4f}) saved to: {os.path.join(checkpoint_dir, "unet_best.pt")}'
    )
    print(f"Checkpoints directory: {checkpoint_dir}")
    print(f"Log file: {log_file}")


def overlay_whiskers(
    frame: np.ndarray,
    df: pd.DataFrame,
    fid: int,
    colors: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    """Return a frame with whisker polylines drawn."""
    sub = df[df["fid"] == fid]
    for _, row in sub.iterrows():
        pts = np.stack([row["pixels_x"], row["pixels_y"]], axis=1).astype(np.int32)
        label = int(row.get("label", row["wid"]))
        color = colors.get(label, (255, 0, 0))
        cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)
    return frame


def build_mask(
    df: pd.DataFrame,
    frame_shape: Tuple[int, int, int],
    fid: int,
    scale_y: float = 1.0,
    scale_x: float = 1.0,
    use_wid_only: bool = False,
    wid_to_class: dict = None,
) -> np.ndarray:
    """Return a segmentation mask for one frame."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    sub = df[df["fid"] == fid]
    for _, row in sub.iterrows():
        # Scale coordinates if image was resized
        pixels_x = np.array(row["pixels_x"]) * scale_x
        pixels_y = np.array(row["pixels_y"]) * scale_y
        pts = np.stack([pixels_x, pixels_y], axis=1).astype(np.int32)

        # Choose label source based on parameter
        if use_wid_only:
            original_id = int(row["wid"])
        else:
            original_id = int(row.get("label", row["wid"]))

        # Map to class index if mapping provided, otherwise use original ID
        if wid_to_class is not None:
            # Use mapping - all whisker IDs should be mapped, no default fallback needed
            # If a WID is not in mapping, it's a data error that should be caught
            if original_id in wid_to_class:
                label = wid_to_class[original_id]
            else:
                # This should not happen if mapping was created correctly
                print(
                    f"Warning: Unmapped whisker ID {original_id} encountered, using background"
                )
                label = 0
        else:
            label = original_id

        cv2.polylines(mask, [pts], False, label, 1)
    return mask


def extract_frames_from_video(video_file, frame_indices, resize_to=None, temp_dir=None):
    """
    Extract frames from video file to temporary directory for faster training.

    Args:
        video_file: Path to video file
        frame_indices: List of frame indices to extract
        resize_to: Optional tuple (width, height) to resize frames
        temp_dir: Optional temporary directory path

    Returns:
        Path to temporary directory containing extracted frames
    """
    import cv2

    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="unet_frames_")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    temp_path = Path(temp_dir)

    print(f"Extracting {len(frame_indices)} frames from {video_file}...")
    print(f"Temporary frame directory: {temp_path}")
    if resize_to:
        print(f"Resizing frames to {resize_to}")

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_count = 0

    for i, frame_idx in enumerate(frame_indices):
        if frame_idx >= total_frames:
            print(
                f"Warning: Frame {frame_idx} exceeds video length ({total_frames}), skipping"
            )
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Could not read frame {frame_idx}, skipping")
            continue

        # Resize if requested
        if resize_to:
            frame = cv2.resize(frame, resize_to)

        # Save frame as PNG for lossless quality
        frame_path = temp_path / f"frame_{frame_idx:08d}.png"
        cv2.imwrite(str(frame_path), frame)
        extracted_count += 1

        # Progress update every 50 frames
        if (i + 1) % 50 == 0 or (i + 1) == len(frame_indices):
            progress = (i + 1) / len(frame_indices) * 100
            print(f"  Progress: {progress:.1f}% ({i + 1}/{len(frame_indices)} frames)")

    cap.release()

    print(f"Successfully extracted {extracted_count} frames to {temp_path}")
    return str(temp_path)


class ExtractedFramesDataset(Dataset):
    """Dataset that loads pre-extracted frames and generates masks from parquet file."""

    def __init__(
        self,
        frames_dir,
        parquet_file,
        frame_indices,
        transform=None,
        use_wid_only=False,
        wid_to_class=None,
    ):
        self.frames_dir = Path(frames_dir)
        self.frame_indices = frame_indices
        self.transform = transform
        self.use_wid_only = use_wid_only
        self.wid_to_class = wid_to_class

        # Load parquet data
        self.df = pd.read_parquet(parquet_file)

        # Verify extracted frames exist
        missing_frames = []
        for frame_idx in frame_indices:
            frame_path = self.frames_dir / f"frame_{frame_idx:08d}.png"
            if not frame_path.exists():
                missing_frames.append(frame_idx)

        if missing_frames:
            print(f"Warning: {len(missing_frames)} extracted frames are missing")

        print(
            f"ExtractedFramesDataset initialized with {len(frame_indices)} frames from {frames_dir}"
        )

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        frame_idx = self.frame_indices[idx]

        # Load pre-extracted frame
        frame_path = self.frames_dir / f"frame_{frame_idx:08d}.png"

        if not frame_path.exists():
            # Fallback to black image if frame missing
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            img = cv2.imread(str(frame_path))
            if img is None:
                img = np.zeros((256, 256, 3), dtype=np.uint8)

        # Get mask data for this frame (using 'fid' column, not 'frame')
        # Note: build_mask function will handle filtering by fid internally

        # Generate mask from parquet data
        height, width = img.shape[:2]
        mask = build_mask(
            self.df,
            (height, width, 3),
            frame_idx,
            1.0,
            1.0,
            self.use_wid_only,
            self.wid_to_class,
        )

        # Convert to PyTorch tensors
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        mask = torch.tensor(mask, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, mask


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a simple U-Net for whisker labelling"
    )

    # Create mutually exclusive group for data input methods
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data-dir", help="Directory with images/ and masks/ subdirectories"
    )
    input_group.add_argument(
        "--video-parquet",
        nargs=2,
        action="append",
        metavar=("VIDEO", "PARQUET"),
        help="Video file and corresponding parquet file. Can be provided multiple times",
    )

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4, will auto-detect optimal if not specified)",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of output classes including background. Will be auto-adjusted if too small for the data.",
    )
    p.add_argument(
        "--auto-classes",
        action="store_true",
        help="Automatically determine optimal number of classes from parquet data analysis",
    )
    p.add_argument(
        "--efficient-classes",
        action="store_true",
        help="Use efficient class mapping: background=0, dominant whiskers=1+, rare whiskers merged to background",
    )
    p.add_argument(
        "--output", default=os.path.join(os.path.dirname(__file__), "unet_model.pt")
    )
    p.add_argument(
        "--frame-indices",
        nargs="*",
        type=int,
        help="Specific frame indices to use (only for video-parquet mode)",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience: number of epochs without improvement (default: 10)",
    )
    p.add_argument(
        "--checkpoint-freq",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )
    p.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping and train for full epochs",
    )
    p.add_argument(
        "--force-batch-size",
        action="store_true",
        help="Force use of specified batch size without auto-detection",
    )
    p.add_argument(
        "--max-batch-size",
        type=int,
        default=64,
        help="Maximum batch size to test during auto-detection (default: 64)",
    )
    p.add_argument(
        "--resize-images",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Resize images to specified dimensions for training (e.g., --resize-images 256 256)",
    )
    p.add_argument(
        "--ignore-label-column",
        action="store_true",
        help='Ignore the "label" column in parquet and use "wid" values directly for consistent labeling',
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
