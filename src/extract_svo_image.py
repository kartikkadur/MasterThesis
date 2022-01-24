import os
import argparse
from videoreaders import SVOReader

def extract(vid_fname, output_dir, save_freq=10, out_fmt="frames"):
    print(f"Extracting: {vid_fname} . . .")
    reader = SVOReader(vid_fname, outdir=output_dir, output=out_fmt)
    for i in range(len(reader)):
        frame = reader.get_frame()
        if i % save_freq == 0:
            reader.write(frame, i)

def run(vid_files, output_dir, save_freq=10, out_fmt='frames'):
    if isinstance(vid_files, list):
        for fn in vid_files:
            extract(fn, output_dir, save_freq, out_fmt)
    else:
        extract(vid_files, output_dir, save_freq, out_fmt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SVO Image extractor")
    parser.add_argument("--vid_fname", type=str, nargs="+", required=True, help="Video filename you want to be extracted")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "outputs"), help="path to output dir")
    parser.add_argument("--output_format", type=str, default="image", help="one of 'image' or 'video' to be saved")
    parser.add_argument("--save_freq", type=int, default=10, help="frequency in which the images are to be saved")
    args = parser.parse_args()
    print(f"Extracting to : {args.output_dir}")
    run(args.vid_fname, args.output_dir, args.save_freq, args.output_format)
            
