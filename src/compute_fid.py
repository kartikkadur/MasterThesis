import torch
import sys
from metrics.fid import compute_fid_from_dirs

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(device)
    #from cleanfid import fid
    dir1, dir2 = sys.argv[1], sys.argv[2]
    # = fid.compute_fid(dir1, dir2, mode='clean')
    score = compute_fid_from_dirs(dir1, dir2, device)
    print(f"FID: {score}")