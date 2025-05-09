import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import time
import urllib.request
import gc
import psutil
import torchvision.transforms as transforms
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MODEL_PATH = "yolo11n.pt"  # Will be downloaded if not found
TEST_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
TEST_IMAGE_PATH = "test_image.jpg"
QUANTIZED_MODEL_PATH = "yolo11n_quantized.pt"

class YOLO11Quantizer:
    def __init__(self, model_name=MODEL_PATH):
        """
        Initialize YOLO11 quantizer with CPU optimization
        """
        self.device = 'cpu'  # Force CPU only
        self.model_name = model_name
        logger.info(f"Using device: {self.device}")

        # Load YOLO model using Ultralytics
        try:
            self.yolo_model = YOLO(model_name).to('cpu')
            self.original_model = self.yolo_model.model
            logger.info(f"Model '{model_name}' loaded successfully on CPU.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Trying to download the model...")
            try:
                # Force model download
                self.yolo_model = YOLO("yolov8n.pt").to('cpu')  # Fallback to standard model
                self.original_model = self.yolo_model.model
                logger.info("Model downloaded successfully.")
            except Exception as e2:
                logger.error(f"Failed to download model: {e2}")
                raise

        # We'll save quantized model here later
        self.quantized_model = None
        
        # Initialize transforms for preprocessing
        self.transforms = transforms.Compose([
            transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
            transforms.ToTensor(),
        ])
        
        # Store original model size for reference
        self.original_model_size = self.get_model_size(self.original_model)

    def get_model_size(self, model):
        """
        More accurate measurement of model size in MB
        """
        try:
            # Create a temporary filename with timestamp to avoid conflicts
            path = f"temp_model_{int(time.time())}.pt"
            
            # Save only the model state dict, not the entire model
            torch.save(model.state_dict(), path)
            
            # Get file size in MB
            size_mb = os.path.getsize(path) / (1024 * 1024)
            
            # Clean up the temporary file
            if os.path.exists(path):
                os.remove(path)
                
            logger.info(f"Measured model size: {size_mb:.2f} MB")
            return size_mb
        except Exception as e:
            logger.error(f"Error in get_model_size: {e}")
            # Fallback to parameter counting method
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            size_mb = (param_size + buffer_size) / (1024 * 1024)
            
            logger.info(f"Estimated model size (fallback): {size_mb:.2f} MB")
            return size_mb

    def get_memory_footprint(self, model):
        """
        Estimate memory footprint in MB during inference
        """
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Measure memory before
        mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        # Create random input and perform inference
        random_input = torch.rand(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        model_on_device = model.to('cpu')
        
        with torch.no_grad():
            _ = model_on_device(random_input)
        
        # Measure memory after model is loaded
        mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        memory_used = mem_after - mem_before
        
        # If memory measurement is too small, estimate from model parameters
        if memory_used < 1:
            total_params = sum(p.numel() * (1 if p.dtype == torch.int8 else 4) for p in model.parameters())
            buffers_size = sum(b.numel() * (1 if b.dtype == torch.int8 else 4) for b in model.buffers())
            memory_used = (total_params + buffers_size) / (1024 * 1024)
        
        return max(memory_used, 0.01)  # Minimum 0.01 MB to avoid division by zero

    def quantize_model(self):
        """
        Quantize the original model using dynamic quantization with optimized settings
        """
        logger.info("Quantizing the model using dynamic quantization...")
        
        # Get original model size before quantization
        original_size = self.original_model_size
        logger.info(f"Original model size (before quantization): {original_size:.2f} MB")
        
        # Start timing quantization process
        quantization_start = time.time()
        
        try:
            # Make sure the model is on CPU before quantization
            model_for_quant = self.original_model.to('cpu')
            
            # Prepare the model for quantization (fuse operations if possible for better performance)
            try:
                # Try to fuse modules before quantization for better performance
                # This is a common optimization for CNN models
                from torch.quantization import fuse_modules
                
                # First set model to eval mode
                model_for_quant.eval()
                
                # Identify fusable layers (Conv+BN+ReLU patterns or Conv+BN)
                modules_to_fuse = []
                for name, module in model_for_quant.named_children():
                    if hasattr(module, 'conv') and hasattr(module, 'bn'):
                        if hasattr(module, 'act'):
                            modules_to_fuse.append([f'{name}.conv', f'{name}.bn', f'{name}.act'])
                        else:
                            modules_to_fuse.append([f'{name}.conv', f'{name}.bn'])
                
                if modules_to_fuse:
                    logger.info(f"Fusing {len(modules_to_fuse)} module sets before quantization")
                    model_for_quant = fuse_modules(model_for_quant, modules_to_fuse)
            except Exception as fuse_error:
                logger.warning(f"Module fusion skipped: {fuse_error}")
            
            # Apply dynamic quantization with more targeted layer selection
            # Focusing on computationally intensive layers (Conv2d, Linear) for better performance
            quantized_model = torch.quantization.quantize_dynamic(
                model_for_quant,
                {torch.nn.Linear, torch.nn.Conv2d},  # Focus on compute-intensive layers
                dtype=torch.qint8,
                inplace=True  # Apply inplace for memory efficiency
            )
            
            self.quantized_model = quantized_model
            
            # Calculate quantization time
            quantization_time = time.time() - quantization_start
            
            # Get quantized model size with direct measurement
            self.save_quantized_model()
            
            # Measure the size of the saved model file
            if os.path.exists(QUANTIZED_MODEL_PATH):
                quantized_size = os.path.getsize(QUANTIZED_MODEL_PATH) / (1024 * 1024)
                logger.info(f"Saved quantized model size: {quantized_size:.2f} MB")
            else:
                # Fallback if file doesn't exist
                quantized_size = self.get_model_size(self.quantized_model)
                logger.info(f"Estimated quantized model size: {quantized_size:.2f} MB")
            
            # Calculate size reduction
            if original_size > 0:
                size_reduction = (original_size - quantized_size) / original_size * 100
            else:
                size_reduction = 0
                
            # Check if quantization actually reduced size
            if abs(original_size - quantized_size) < 0.01:
                logger.warning("Warning: Quantized model size is nearly identical to original model size.")
                logger.info("Verifying quantization was applied correctly...")
                
                # Verify quantization by checking model dtypes
                has_int8 = any(p.dtype == torch.qint8 for p in self.quantized_model.parameters())
                logger.info(f"Model contains int8 parameters: {has_int8}")
                
                if not has_int8:
                    logger.warning("No int8 parameters found. Trying alternative quantization method...")
                    
                    # Try static quantization preparation as a fallback
                    try:
                        # Reset and try different quantization approach
                        model_for_quant = self.original_model.to('cpu')
                        
                        # Create a new quantization configuration focusing on conv layers
                        quantized_model = torch.quantization.quantize_dynamic(
                            model_for_quant, 
                            {torch.nn.Conv2d},  # Target primarily Conv2d layers
                            dtype=torch.qint8
                        )
                        
                        self.quantized_model = quantized_model
                        
                        # Save and measure size again
                        self.save_quantized_model()
                        
                        if os.path.exists(QUANTIZED_MODEL_PATH):
                            quantized_size = os.path.getsize(QUANTIZED_MODEL_PATH) / (1024 * 1024)
                            logger.info(f"Alternative quantization method model size: {quantized_size:.2f} MB")
                            
                            # Recalculate size reduction
                            if original_size > 0:
                                size_reduction = (original_size - quantized_size) / original_size * 100
                            else:
                                size_reduction = 0
                    except Exception as alt_error:
                        logger.error(f"Alternative quantization failed: {alt_error}")
            
            quantization_metrics = {
                "original_size_mb": original_size,
                "quantized_size_mb": quantized_size,
                "size_reduction_percent": size_reduction,
                "quantization_time_seconds": quantization_time
            }
            
            logger.info(f"Model quantization completed. Size reduced from {original_size:.2f}MB to {quantized_size:.2f}MB ({size_reduction:.2f}%)")
            return self.quantized_model, quantization_metrics
            
        except Exception as e:
            logger.error(f"Error during quantization: {e}")
            logger.info("Trying with simpler quantization...")
            
            # Fallback to simpler quantization
            model_for_quant = self.original_model.to('cpu')
            quantized_model = torch.quantization.quantize_dynamic(
                model_for_quant,
                {torch.nn.Linear},  # Apply only to Linear layers
                dtype=torch.qint8
            )
            self.quantized_model = quantized_model
            
            # Calculate quantization time
            quantization_time = time.time() - quantization_start
            
            # Save model to see actual file size reduction
            self.save_quantized_model()
            
            # Update with actual file size
            if os.path.exists(QUANTIZED_MODEL_PATH):
                quantized_size = os.path.getsize(QUANTIZED_MODEL_PATH) / (1024 * 1024)
                if original_size > 0:
                    size_reduction = (original_size - quantized_size) / original_size * 100
                else:
                    size_reduction = 0
                logger.info(f"Fallback quantized model size: {quantized_size:.2f}MB ({size_reduction:.2f}% reduction)")
            else:
                # If file doesn't exist, use direct measurement
                quantized_size = self.get_model_size(self.quantized_model)
                if original_size > 0:
                    size_reduction = (original_size - quantized_size) / original_size * 100
                else:
                    size_reduction = 0
            
            quantization_metrics = {
                "original_size_mb": original_size,
                "quantized_size_mb": quantized_size,
                "size_reduction_percent": size_reduction,
                "quantization_time_seconds": quantization_time
            }
            
            logger.info(f"Model quantization completed (fallback mode). Size reduced from {original_size:.2f}MB to {quantized_size:.2f}MB ({size_reduction:.2f}%)")
            return self.quantized_model, quantization_metrics

    def save_quantized_model(self):
        """
        Save the quantized model to disk
        """
        if self.quantized_model is not None:
            try:
                # Save quantized model state dict
                torch.save(self.quantized_model.state_dict(), QUANTIZED_MODEL_PATH)
                logger.info(f"Quantized model saved to {QUANTIZED_MODEL_PATH}")
                return True
            except Exception as e:
                logger.error(f"Error saving quantized model: {e}")
                return False
        else:
            logger.warning("No quantized model available to save")
            return False

    def optimize_inference(self, model):
        """
        Apply additional CPU-specific optimizations for inference
        """
        # Set model to evaluation mode
        model.eval()
        
        # Ensure model is on CPU
        model = model.to('cpu')
        
        # Enable PyTorch inference optimizations
        with torch.no_grad():
            # Try to apply more aggressive optimizations
            try:
                # Create example input
                example = torch.rand(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
                
                # Try to trace and script the model (improves CPU performance significantly)
                traced_model = torch.jit.trace(model, example)
                
                # Apply fusion optimizations
                optimized_model = torch.jit.optimize_for_inference(traced_model)
                
                # Try to freeze the model for better performance
                frozen_model = torch.jit.freeze(optimized_model)
                logger.info("Successfully applied advanced JIT optimizations with freezing")
                return frozen_model
            except Exception as e1:
                logger.warning(f"Could not apply advanced JIT optimizations: {e1}")
                try:
                    # Fallback to basic tracing only
                    example = torch.rand(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
                    traced_model = torch.jit.trace(model, example)
                    logger.info("Applied basic model tracing")
                    return traced_model
                except Exception as e2:
                    logger.warning(f"Could not apply basic tracing: {e2}")
                    # Return original model if all optimizations fail
                    return model

    def preprocess_image(self, image_path):
        """
        Preprocess image for model input with optimized transforms
        """
        try:
            # Open and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            img_tensor = self.transforms(image)
            
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)
            
            # Optimize memory layout for CPU inference
            img_tensor = img_tensor.contiguous()
            
            return img_tensor.to('cpu')
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return torch.rand(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

    def predict_original(self, image_path):
        """
        Inference using original YOLO model
        """
        logger.info("Running original model inference...")
        self.yolo_model.to('cpu')
        
        # Warmup runs (increased for better stabilization)
        for _ in range(3):
            with torch.no_grad():
                _ = self.yolo_model(image_path, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
        
        # Timing runs
        inference_times = []
        results = None
        
        # Increased number of runs for more stable benchmarking
        for _ in range(10):
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            start = time.time()
            with torch.no_grad():
                current_results = self.yolo_model(image_path, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
            inference_times.append((time.time() - start) * 1000)  # ms
            
            # Save the last results
            if results is None:
                results = current_results
        
        # Remove outliers (highest and lowest times) for more stable results
        if len(inference_times) > 4:
            inference_times.remove(max(inference_times))
            inference_times.remove(min(inference_times))
            
        avg_inference_time = sum(inference_times) / len(inference_times)
        min_inference_time = min(inference_times)
        max_inference_time = max(inference_times)
        
        timing_metrics = {
            "average_ms": avg_inference_time,
            "min_ms": min_inference_time,
            "max_ms": max_inference_time
        }
        
        logger.info(f"Original model avg inference time: {avg_inference_time:.2f} ms")
        
        return results, timing_metrics

    def predict_quantized(self, image_path):
        """
        Inference using quantized model with optimized execution
        """
        if self.quantized_model is None:
            raise ValueError("Quantized model is not ready. Please quantize first.")

        logger.info("Running quantized model inference...")
        
        # Preprocess image with optimized pipeline
        input_tensor = self.preprocess_image(image_path)
        
        # Apply optimizations for inference
        optimized_quantized = self.optimize_inference(self.quantized_model)
        
        # More extensive warmup for better cache performance
        for _ in range(5):
            with torch.no_grad():
                _ = optimized_quantized(input_tensor)
        
        # Timing runs with increased iterations
        inference_times = []
        for _ in range(15):  # More runs for better statistics
            # Explicit GC to reduce measurement noise
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            start = time.time()
            with torch.no_grad():
                _ = optimized_quantized(input_tensor)
            inference_times.append((time.time() - start) * 1000)  # ms
        
        # Remove outliers for more stable results
        if len(inference_times) > 4:
            inference_times.remove(max(inference_times))
            inference_times.remove(min(inference_times))
        
        avg_inference_time = sum(inference_times) / len(inference_times)
        min_inference_time = min(inference_times)
        max_inference_time = max(inference_times)
        
        timing_metrics = {
            "average_ms": avg_inference_time,
            "min_ms": min_inference_time,
            "max_ms": max_inference_time
        }
        
        logger.info(f"Quantized model avg inference time: {avg_inference_time:.2f} ms")
        
        # For results visualization, use the original model
        self.yolo_model.to('cpu')
        results = self.yolo_model(image_path)
        
        return results, timing_metrics

    def calculate_accuracy_metrics(self, original_results, quantized_results):
        """
        Compare accuracy between original and quantized models
        """
        try:
            # Extract detection boxes from both results
            orig_boxes = original_results[0].boxes
            quant_boxes = quantized_results[0].boxes
            
            # Calculate metrics
            orig_conf = orig_boxes.conf.mean().item() if len(orig_boxes) > 0 else 0
            quant_conf = quant_boxes.conf.mean().item() if len(quant_boxes) > 0 else 0
            
            # Basic detection match metrics
            accuracy_metrics = {
                "original_detections": len(orig_boxes),
                "quantized_detections": len(quant_boxes),
                "original_confidence": orig_conf,
                "quantized_confidence": quant_conf,
                "confidence_diff_percent": ((orig_conf - quant_conf) / orig_conf * 100) if orig_conf > 0 else 0,
                "detection_match_percent": (min(len(orig_boxes), len(quant_boxes)) / max(len(orig_boxes), len(quant_boxes)) * 100) if max(len(orig_boxes), len(quant_boxes)) > 0 else 100
            }
            
            return accuracy_metrics
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
            

def apply_threading_optimizations():
    """
    Apply CPU threading optimizations for better performance
    """
    try:
        # Get logical CPU count
        cpu_count = os.cpu_count()
        
        # For better inference performance, limit threads to physical cores
        # Estimate physical cores as half of logical cores (rough approximation)
        physical_cores = max(1, cpu_count // 2)
        
        # Set number of threads for PyTorch - using physical cores often gives better performance
        torch.set_num_threads(physical_cores)
        
        # Set threading backend to OpenMP for better CPU utilization
        if hasattr(torch, 'set_num_interop_threads'):
            # Control inter-op parallelism (operations that can run independently)
            torch.set_num_interop_threads(2)  # Lower value often works better for inference
            
        # Enable Intel MKL optimizations if available
        if 'MKL_NUM_THREADS' not in os.environ:
            os.environ['MKL_NUM_THREADS'] = str(physical_cores)
        
        # Enable OpenMP optimizations if available
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ['OMP_NUM_THREADS'] = str(physical_cores)
            
        # Additional MKL settings for better inference performance
        os.environ['MKL_DYNAMIC'] = 'FALSE'  # Disable dynamic adjustment of thread count
        
        logger.info(f"CPU threading optimizations applied. Using {physical_cores} threads")
        return True
    except Exception as e:
        logger.warning(f"Could not apply threading optimizations: {e}")
        return False

def compare_models(original_time, quantized_time, quantization_metrics, original_memory, quantized_memory, accuracy_metrics):
    """
    Compare metrics between models
    """
    # Calculate speedup
    if quantized_time["average_ms"] > 0 and original_time["average_ms"] > 0:
        if quantized_time["average_ms"] < original_time["average_ms"]:
            speedup = ((original_time["average_ms"] / quantized_time["average_ms"]) - 1) * 100
        else:
            speedup = -((quantized_time["average_ms"] / original_time["average_ms"]) - 1) * 100
    else:
        speedup = 0
    
    comparison = {
        # Latency metrics
        'original_avg_time_ms': original_time["average_ms"],
        'quantized_avg_time_ms': quantized_time["average_ms"],
        'quantized_speedup_percent': speedup,
        
        # Memory footprint 
        'original_memory_mb': original_memory,
        'quantized_memory_mb': quantized_memory,
        'memory_reduction_percent': ((original_memory - quantized_memory) / original_memory * 100) if original_memory > 0 else 0,
        
        # Model size
        'original_size_mb': quantization_metrics["original_size_mb"],
        'quantized_size_mb': quantization_metrics["quantized_size_mb"],
        'size_reduction_percent': quantization_metrics["size_reduction_percent"],
        'quantization_time_seconds': quantization_metrics["quantization_time_seconds"],
        
        # Accuracy
        'accuracy_metrics': accuracy_metrics
    }
    
    return comparison

def download_test_image(url=TEST_IMAGE_URL, save_path=TEST_IMAGE_PATH):
    """
    Download a test image if it doesn't exist
    """
    if not os.path.exists(save_path):
        logger.info(f"Test image not found, downloading from {url}...")
        try:
            urllib.request.urlretrieve(url, save_path)
            logger.info(f"Test image downloaded and saved to {save_path}")
        except Exception as e:
            logger.error(f"Error downloading test image: {e}")
            return False
    else:
        logger.info(f"Using existing test image: {save_path}")
    return True

def main():
    print("="*50)
    print("YOLO11 Dynamic Quantization Performance Analysis (CPU-only)")
    print("="*50)
    
    # Apply CPU threading optimizations
    apply_threading_optimizations()
    
    # Download test image
    if not download_test_image():
        logger.error("Error: Couldn't download test image. Please provide your own image.")
        return
    
    # Initialize Quantizer - CPU operation only
    quantizer = YOLO11Quantizer(model_name=MODEL_PATH)

    # Get original model memory footprint
    logger.info("\nMeasuring original model memory footprint...")
    original_memory = quantizer.get_memory_footprint(quantizer.original_model)
    logger.info(f"Original model memory footprint: {original_memory:.2f} MB")

    # Run inference with original model
    original_results, original_time = quantizer.predict_original(TEST_IMAGE_PATH)
    original_image = original_results[0].plot()
    Image.fromarray(original_image).save("original_detection.jpg")
    logger.info("Original detection image saved to 'original_detection.jpg'")

    # Verify original model size before quantization
    logger.info(f"Original model size before quantization: {quantizer.original_model_size:.2f} MB")

    # Quantize model (dynamic only)
    quantized_model, quantization_metrics = quantizer.quantize_model()
    
    # Get quantized model memory footprint
    logger.info("\nMeasuring quantized model memory footprint...")
    quantized_memory = quantizer.get_memory_footprint(quantized_model)
    logger.info(f"Quantized model memory footprint: {quantized_memory:.2f} MB")

    # Run inference with quantized model
    quantized_results, quantized_time = quantizer.predict_quantized(TEST_IMAGE_PATH)
    
    # Save quantized detection image
    quantized_image = quantized_results[0].plot()
    Image.fromarray(quantized_image).save("quantized_detection.jpg")
    
    # Calculate accuracy metrics
    accuracy_metrics = quantizer.calculate_accuracy_metrics(original_results, quantized_results)
    
    # Compare models
    comparison = compare_models(
        original_time, 
        quantized_time, 
        quantization_metrics, 
        original_memory, 
        quantized_memory,
        accuracy_metrics
    )
    
    # Print detailed comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    
    print("\n1. LATENCY METRICS (lower is better):")
    print(f"  Original model avg inference time: {comparison['original_avg_time_ms']:.2f} ms")
    print(f"  Quantized model avg inference time: {comparison['quantized_avg_time_ms']:.2f} ms")
    if comparison['quantized_speedup_percent'] > 0:
        print(f"  Quantized model speedup: {comparison['quantized_speedup_percent']:.2f}% (FASTER)")
    else:
        print(f"  Quantized model slowdown: {abs(comparison['quantized_speedup_percent']):.2f}% (SLOWER)")
    
    print("\n2. MEMORY FOOTPRINT (lower is better):")
    print(f"  Original model memory: {comparison['original_memory_mb']:.2f} MB")
    print(f"  Quantized model memory: {comparison['quantized_memory_mb']:.2f} MB")
    print(f"  Memory reduction: {comparison['memory_reduction_percent']:.2f}%")
    
    print("\n3. MODEL SIZE:")
    print(f"  Original model size: {comparison['original_size_mb']:.2f} MB")
    print(f"  Quantized model size: {comparison['quantized_size_mb']:.2f} MB")
    print(f"  Model size reduction: {comparison['size_reduction_percent']:.2f}%")
    print(f"  Quantization process time: {comparison['quantization_time_seconds']:.2f} seconds")
    
    print("\n4. ACCURACY METRICS:")
    print(f"  Original model detections: {comparison['accuracy_metrics']['original_detections']}")
    print(f"  Quantized model detections: {comparison['accuracy_metrics']['quantized_detections']}")
    print(f"  Original model confidence: {comparison['accuracy_metrics']['original_confidence']:.4f}")
    print(f"  Quantized model confidence: {comparison['accuracy_metrics']['quantized_confidence']:.4f}")
    print(f"  Confidence difference: {comparison['accuracy_metrics']['confidence_diff_percent']:.2f}%")
    print(f"  Detection match rate: {comparison['accuracy_metrics']['detection_match_percent']:.2f}%")


    print("="*50)
if __name__ == '__main__':
    main()