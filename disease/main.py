import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Choose training script to run.")
    parser.add_argument(
        "--mode", choices=["raw", "masked"], required=True,
        help="Select 'raw' for original image training, or 'masked' for masked image training"
    )
    args = parser.parse_args()

    if args.mode == "raw":
        subprocess.run(["python", "-m", "scripts.train.train_classify_diseases"])
    elif args.mode == "masked":
        subprocess.run(["python", "-m", "scripts.train.train_masked_model"])
    elif args.mode == "new_data":
        subprocess.run(["python", "-m", "scripts.train.new_data_train"])

if __name__ == "__main__":
    main()
