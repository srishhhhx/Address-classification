"""
Download IndicBERT model weights from Google Drive

This script downloads the model.safetensors file from Google Drive
and places it in the correct directory for the IndicBERT classifier.

Usage:
    python download_model.py
"""

import os
import sys

def print_instructions():
    """Print download instructions"""
    print("=" * 80)
    print("IndicBERT Model Download Instructions")
    print("=" * 80)
    print()
    print("The model weights (model.safetensors, 1.1 GB) are hosted on Google Drive.")
    print()
    print("OPTION 1: Manual Download (Recommended)")
    print("-" * 80)
    print("1. Visit the Google Drive link:")
    print("   https://drive.google.com/drive/folders/1zXUqzC42vWChcqTVmW35OMgqJmABYPCC?usp=sharing")
    print()
    print("2. Download the file 'model.safetensors' (1.1 GB)")
    print()
    print("3. Place it in this directory:")
    print(f"   {os.path.dirname(os.path.abspath(__file__))}/")
    print()
    print("4. Verify the file is in the correct location:")
    print(f"   Expected path: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.safetensors')}")
    print()
    print("-" * 80)
    print()
    print("OPTION 2: Use gdown (Automatic Download)")
    print("-" * 80)
    print("If you have gdown installed, you can use:")
    print()
    print("  pip install gdown")
    print("  gdown [FILE_ID] -O model.safetensors")
    print()
    print("Where [FILE_ID] is extracted from the Google Drive sharing link.")
    print()
    print("-" * 80)
    print()
    print("OPTION 3: View Results Without Downloading")
    print("-" * 80)
    print("All model results are pre-computed in:")
    print("  testing/indicbert_testing.ipynb")
    print()
    print("You can review all metrics, visualizations, and predictions")
    print("without downloading the model.")
    print()
    print("=" * 80)

def check_model_exists():
    """Check if model.safetensors already exists"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.safetensors")

    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        file_size_gb = file_size / (1024**3)

        print()
        print("✅ Model file found!")
        print(f"   Path: {model_path}")
        print(f"   Size: {file_size_gb:.2f} GB")
        print()

        if file_size_gb < 1.0:
            print("⚠️  Warning: File size seems small. Expected ~1.1 GB.")
            print("   The download may be incomplete.")
        else:
            print("✅ File size looks correct. Model should be ready to use!")

        print()
        return True
    else:
        print()
        print("❌ Model file not found.")
        print(f"   Expected location: {model_path}")
        print()
        return False

def main():
    print()

    # Check if model exists
    model_exists = check_model_exists()

    if not model_exists:
        print_instructions()
        print()
        print("After downloading, run this script again to verify the file.")
    else:
        print("Next steps:")
        print("1. Run the testing notebook: testing/indicbert_testing.ipynb")
        print("2. Or use the model in your own code:")
        print()
        print("   from transformers import AutoTokenizer, AutoModelForSequenceClassification")
        print()
        print("   MODEL_PATH = 'best_models/indicbert_property_classifier'")
        print("   tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)")
        print("   model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)")
        print()

if __name__ == "__main__":
    main()
