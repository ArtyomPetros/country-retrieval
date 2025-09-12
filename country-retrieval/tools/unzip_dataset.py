
import argparse, zipfile, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True, help='Path to dataset zip')
    ap.add_argument('--out', required=True, help='Output directory')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with zipfile.ZipFile(args.zip, 'r') as zf:
        zf.extractall(args.out)
    print(f"Unzipped to {args.out}")

if __name__ == "__main__":
    main()
