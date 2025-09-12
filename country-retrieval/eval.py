
import os, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import DocDataset
from model import Embedder

@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs, ys = [], []
    for x, y in tqdm(loader, desc="Extract"):
        x = x.to(device, non_blocking=True)
        e = model(x).cpu()
        embs.append(e)
        ys.append(y)
    return torch.cat(embs, 0), torch.cat(ys, 0).numpy()

def metrics(embs, ys, protos):
    sims = embs @ protos.T  # [N,C]
    top1 = sims.argmax(1).numpy()
    acc1 = (top1 == ys).mean()

    # Top-3
    top3 = np.argsort(-sims.numpy(), axis=1)[:, :3]
    acc3 = np.mean([ys[i] in top3[i] for i in range(len(ys))])

    # mAP@5
    top5 = np.argsort(-sims.numpy(), axis=1)[:, :5]
    ap = []
    for i in range(len(ys)):
        hits = (top5[i] == ys[i]).astype(float)
        if hits.sum() == 0:
            ap.append(0.0)
        else:
            # precision at k where hit occurs
            k = np.where(hits==1)[0][0] + 1
            ap.append(1.0/k)
    map5 = float(np.mean(ap))
    return acc1, acc3, map5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--prototypes', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--img-size', type=int, default=256)
    ap.add_argument('--split', choices=['train','val','all'], default='val')
    args = ap.parse_args()

    device = torch.device('cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')

    ckpt = torch.load(args.weights, map_location='cpu')
    model = Embedder(embed_dim=ckpt.get('args',{}).get('embed_dim',256))
    model.load_state_dict(ckpt['model'])
    model.to(device)

    # Choice of split simplified: use full data unless you manage explicit splits externally.
    ds = DocDataset(args.data_root, img_size=args.img_size, train=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    embs, ys = extract_embeddings(model, loader, device)
    protos = torch.load(args.prototypes, map_location='cpu')
    embs = torch.nn.functional.normalize(embs, dim=1)
    protos = torch.nn.functional.normalize(protos, dim=1)

    acc1, acc3, map5 = metrics(embs, ys, protos)
    print(f"Recall@1: {acc1:.4f}, Recall@3: {acc3:.4f}, mAP@5: {map5:.4f}")

if __name__ == "__main__":
    main()
