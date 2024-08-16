import argparse
import logging
from ultralytics import YOLO
import os
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ['onnx', 'torchscript', 'coreml', 'tflite', 'tfjs']

def validate_export_format(export_format):
    if export_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported export format: {export_format}. Supported formats are: {', '.join(SUPPORTED_FORMATS)}")

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist.")
    try:
        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def export_model(model, export_format):
    try:
        logger.info(f"Exporting model to {export_format} format")
        model.export(format=export_format)
        logger.info("Model exported successfully")
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        raise

def detect_gpu():
    if torch.cuda.is_available():
        logger.info("NVIDIA GPU detected.")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        logger.info("AMD GPU detected (via MPS).")
        return torch.device('mps')
    elif torch.has_mps:
        logger.info("AMD GPU detected (via ROCm).")
        return torch.device('mps')
    else:
        logger.info("No compatible GPU detected. Using CPU.")
        return torch.device('cpu')

def main():
    parser = argparse.ArgumentParser(description="Load and export YOLO model")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the YOLO model file (e.g., 'best.pt')")
    parser.add_argument('--export-format', type=str, default='onnx', help=f"Export format (default: 'onnx'). Supported formats: {', '.join(SUPPORTED_FORMATS)}")

    args = parser.parse_args()

    try:
        validate_export_format(args.export_format)
        device = detect_gpu()
        model = load_model(args.model_path)
        model.to(device)
        export_model(model, args.export_format)
    except FileNotFoundError as fnf_error:
        logger.error(fnf_error)
    except ValueError as val_error:
        logger.error(val_error)
    except Exception as e:
        logger.error(f"Failed to complete the operation: {e}")

if __name__ == "__main__":
    main()