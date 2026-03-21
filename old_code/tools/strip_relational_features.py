import h5py
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

def strip_dataset(input_path: str, output_path: str):
    """
    Copies an HDF5 dataset to a new file, excluding 'relational_features'.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)
        
    if output_path.exists():
        print(f"Error: Output file {output_path} already exists.")
        sys.exit(1)
        
    print(f"Stripping relational features from {input_path} -> {output_path}")
    
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        # Copy Attributes
        for k, v in src.attrs.items():
            dst.attrs[k] = v
            
        # Copy Datasets
        for key in tqdm(src.keys(), desc="Copying datasets"):
            if key == "relational_features":
                print(f"Skipping {key}...")
                continue
                
            print(f"Copying {key}...")
            # We can use h5py's group copy, but let's be explicit to ensure compression etc if needed.
            # actually src.copy_to(dst, key) is easiest
            src.copy(key, dst)
            
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip relational_features from HDF5 dataset.")
    parser.add_argument("input_path", type=str, help="Path to input HDF5 file")
    parser.add_argument("output_path", type=str, help="Path to output HDF5 file")
    
    args = parser.parse_args()
    
    strip_dataset(args.input_path, args.output_path)
