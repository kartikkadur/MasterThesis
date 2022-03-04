import torch
import sys
from cleanfid import fid

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dir1, dir2 = sys.argv[1], sys.argv[2]
    score = fid.compute_fid(dir1, dir2, mode='clean')
    print(f"FID: {score}")