import os, argparse, json
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--outdir', default='./artifacts')
    ap.add_argument('--img-size', type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device('cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')

    ckpt = torch.load(args.weights, map_location='cpu')
    embed_dim = ckpt.get('args',{}).get('embed_dim',256)
    model = Embedder(embed_dim=embed_dim, pretrained=False)  # инференс без скачивания весов
    model.load_state_dict(ckpt['model'])
    model.to(device)

    ds = DocDataset(args.data_root, img_size=args.img_size, train=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    embs, ys = extract_embeddings(model, loader, device)  # embs: [N,D], ys: [N]
    embs = torch.nn.functional.normalize(embs, dim=1)

    C = len(ds.idx_to_class)
    D = embs.size(1)
    protos = torch.zeros(C, D)
    ys_t = torch.tensor(ys)
    for c in range(C):
        idxs = (ys_t == c).nonzero(as_tuple=True)[0]
        if idxs.numel() > 0:
            protos[c] = embs[idxs].mean(0)
    protos = torch.nn.functional.normalize(protos, dim=1)

    with open(os.path.join(args.outdir, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump({int(k):v for k,v in ds.idx_to_class.items()}, f, ensure_ascii=False, indent=2)
    torch.save(protos, os.path.join(args.outdir, 'prototypes.pt'))
    print("Saved prototypes and labels to", args.outdir)

if __name__ == "__main__":
    main()
