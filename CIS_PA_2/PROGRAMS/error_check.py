import os
import numpy as np

extract_path = r"C:\Users\aarus\Documents\CIS_PA_2\aarushi_output\\"
datasets = os.listdir(extract_path)

def check_output_error(output_file, reference_file, verbose=True):
    """Compare your output vs reference (Output 2 a-f)."""
    def load_output(fname):
        with open(fname, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        data = [list(map(float, ln.split(","))) for ln in lines[1:]]
        return np.array(data)
    
    if not os.path.exists(output_file):
        print(f"⚠️ Output file missing: {output_file}")
        return None
    if not os.path.exists(reference_file):
        print(f"⚠️ Reference file missing: {reference_file}")
        return None
    
    your_data = load_output(output_file)
    ref_data = load_output(reference_file)
    
    if your_data.shape != ref_data.shape:
        print(f"⚠️ Shape mismatch: {your_data.shape} vs {ref_data.shape}")
        return None
    
    errors = np.linalg.norm(your_data - ref_data, axis=1)
    mean_err = errors.mean()
    max_err = errors.max()
    
    if verbose:
        print(f"Processing {os.path.basename(output_file)}:")
        print(f"  Mean error: {mean_err:.3f} mm")
        print(f"  Max error:  {max_err:.3f} mm")
        print(f"  RMS error:  {np.sqrt(np.mean(errors**2)):.3f} mm\n")
    
    return {
        "per_point_errors": errors,
        "mean_error": mean_err,
        "max_error": max_err,
        "rms_error": np.sqrt(np.mean(errors**2))
    }


def main():
    letters_to_process = ["a","b","c","d","e","f"]
    
    for fname in datasets:
        if fname.endswith("-output2.txt"):
            data_prefix = fname[:-len("-output2.txt")]
            
            # Check if dataset letter is in a-f
            letter = data_prefix.split("-")[-1]  # assumes letter is last in prefix
            if letter not in letters_to_process:
                print(f"Skipping dataset {letter}")
                continue
            
            print(f"\n==== Processing dataset {letter} ({data_prefix}) ====")
            
            output_file = os.path.join(extract_path, f"{data_prefix}-myoutput2.txt")
            reference_file = os.path.join(extract_path, f"{data_prefix}-output2.txt")
            check_output_error(output_file, reference_file)

if __name__ == "__main__":
    main()
