import argparse
from src.detector import run_detection

def main():
    ap = argparse.ArgumentParser(description="Video Temporal Error Detector (Drop/Merge)")
    ap.add_argument("--input", required=True, help="Input video path")
    ap.add_argument("--mode", choices=["heuristic","ai"], default="heuristic", help="Detection mode")
    ap.add_argument("--out_dir", default="output", help="Output directory")
    ap.add_argument("--report_dir", default="reports", help="Report directory")
    ap.add_argument("--resize_w", type=int, default=320, help="Analysis resize width")
    ap.add_argument("--resize_h", type=int, default=180, help="Analysis resize height")
    ap.add_argument("--max_frames", type=int, default=0, help="Process only first N frames (0=all)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (AI mode)")
    args = ap.parse_args()

    run_detection(
        input_path=args.input,
        mode=args.mode,
        out_dir=args.out_dir,
        report_dir=args.report_dir,
        resize=(args.resize_w, args.resize_h),
        max_frames=(None if args.max_frames<=0 else args.max_frames),
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
