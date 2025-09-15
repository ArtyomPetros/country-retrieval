import argparse, json, time
import torch
from PIL import Image
import numpy as np
from model import Embedder
from data_utils import build_tfms

def load_image(path, img_size):
    img = Image.open(path).convert('RGB')
    img = np.array(img)
    tfm = build_tfms(img_size, train=False)
    x = tfm(image=img)['image'].unsqueeze(0)
    return x

def pick_device(pref: str):
    if pref == 'cpu':
        return torch.device('cpu')
    if pref == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    if pref == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--prototypes', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--img-size', type=int, default=256)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    ap.add_argument('--print-prec', type=int, default=4)
    ap.add_argument('--warmup', type=int, default=1, help='warmup iterations to stabilize timing')
    args = ap.parse_args()

    device = pick_device(args.device)

    ckpt = torch.load(args.weights, map_location='cpu')
    model = Embedder(embed_dim=ckpt.get('args',{}).get('embed_dim',256), pretrained=False)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    protos = torch.load(args.prototypes, map_location='cpu')
    with open(args.labels, 'r', encoding='utf-8') as f:
        idx_to_class = {int(k):v for k,v in json.load(f).items()}

    x = load_image(args.image, args.img_size).to(device)
    with torch.no_grad():
        for _ in range(max(0, args.warmup)):
            _ = model(x)
    t0 = time.time()
    with torch.no_grad():
        e = model(x).cpu()
        sims = (e @ protos.T).squeeze(0)
        vals, idxs = torch.topk(sims, k=min(args.topk, protos.size(0)))
    dt_ms = (time.time() - t0) * 1000

    top = [(idx_to_class[int(i)], float(v)) for v,i in zip(vals.tolist(), idxs.tolist())]
    print(f"Inference time (model+sim): {dt_ms:.1f} ms on {device}")
    print("Top-K:")
    for name, score in top:
        print(f"{name}: {score:.{args.print_prec}f}")

if __name__ == "__main__":
    main()
