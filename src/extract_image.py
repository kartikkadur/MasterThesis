import os
import argparse
from videoreaders import SVOReader

def extract(vid_fname, output_dir, save_freq=10, out_fmt="frames"):
    print(f"Extracting: {vid_fname} . . .")
    reader = SVOReader(vid_fname, outdir=output_dir, outfmt=out_fmt)
    for i in range(len(reader)):
        frame = reader.get_frame()
        if i % save_freq == 0:
            reader.write(frame, i)

def run(dataroot, output_dir, save_freq=10, out_fmt='frames'):
    if os.path.isdir(dataroot):
        vfiles = [os.path.join(dataroot, vfile) for vfile in os.listdir(dataroot)]
        for vfile in vfiles:
            extract(vfile, output_dir, save_freq, out_fmt)
    else:
        extract(dataroot, output_dir, save_freq, out_fmt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SVO Image extractor")
    parser.add_argument("--dataroot", type=str, required=True, help="a video file or path to a directory containing video files")
    parser.add_argument("--outdir", type=str, default=os.path.join(os.getcwd(), "outputs"), help="path to output dir")
    parser.add_argument("--outfmt", type=str, default="image", help="one of 'image' or 'video' to be saved")
    parser.add_argument("--save_freq", type=int, default=10, help="frequency in which the images are to be saved")
    args = parser.parse_args()
    print(f"Extracting to : {args.outdir}")
    run(args.dataroot, args.outdir, args.save_freq, args.outfmt)