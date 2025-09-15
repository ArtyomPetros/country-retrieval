
import os, argparse, random, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from model import Embedder
from data_utils import DocDataset, list_images, PKSampler
from model import Embedder
from losses import BatchHardTripletLoss

def set_seed(seed: int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_indices(labels, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    idxs = np.arange(len(labels))
    train_idx, val_idx = [], []
    for y in np.unique(labels):
        cls_idx = idxs[labels==y]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx)*val_ratio))
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])
    return train_idx, val_idx

@torch.no_grad()
def recall_at_1(model, loader, device):
    model.eval()
    embs = []
    ys = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        e = model(x).cpu()
        embs.append(e)
        ys.append(y)
    embs = torch.cat(embs, 0)
    ys = torch.cat(ys, 0).numpy()
    D = torch.cdist(embs, embs, p=2).numpy()
    import numpy as np
    np.fill_diagonal(D, 1e9)
    nn_idx = D.argmin(axis=1)
    pred = ys[nn_idx]
    return (pred == ys).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--outdir', default='./artifacts')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--img-size', type=int, default=256)
    ap.add_argument('--embed-dim', type=int, default=256)
    ap.add_argument('--p-classes', type=int, default=8)
    ap.add_argument('--k-samples', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pretrained', type=int, default=0, help='1=use ImageNet weights (needs internet)')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    # Device: Prefer MPS (Apple GPU), then CUDA, else CPU
    device = torch.device('cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    print(f"Using device: {device}")

    paths, labels, idx_to_class = list_images(args.data_root)
    if len(paths) == 0:
        raise RuntimeError(f"No images found under {args.data_root}")

    train_idx, val_idx = split_indices(labels, val_ratio=0.2, seed=args.seed)
    train_ds = DocDataset(args.data_root, img_size=args.img_size, train=True, indices=train_idx)
    val_ds = DocDataset(args.data_root, img_size=args.img_size, train=False, indices=val_idx)

    sampler = PKSampler([train_ds.all_labels[i] for i in train_idx], args.p_classes, args.k_samples)
    train_loader = DataLoader(train_ds, batch_size=args.p_classes*args.k_samples, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = Embedder(embed_dim=args.embed_dim, pretrained=bool(args.pretrained)).to(device)
    criterion = BatchHardTripletLoss(margin=0.2)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(optim, T_max=args.epochs)

    best_recall = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        t1 = time.time()
        prog = tqdm(range(len(train_loader)), desc=f"Epoch {epoch}/{args.epochs}")
        it = iter(train_loader)
        for _ in prog:
            try:
                x, y = next(it)
            except StopIteration:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optim.zero_grad()
            with torch.autocast(device_type='cuda' if device.type=='cuda' else 'cpu', enabled=(device.type=='cuda')):
                e = model(x)
                loss = criterion(e, y)
            if device.type=='cuda':
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            losses.append(loss.item())
            import numpy as np
            prog.set_postfix(loss=np.mean(losses[-50:]))

        sched.step()
        dur = time.time()-t1
        r1 = recall_at_1(model, val_loader, device)
        print(f"Epoch {epoch}: loss={np.mean(losses):.4f}, val R@1={r1:.4f}, time={dur:.1f}s")
        if r1 > best_recall:
            best_recall = r1
            torch.save({'model': model.state_dict(),
                        'args': vars(args),
                        'idx_to_class': train_ds.idx_to_class}, os.path.join(args.outdir, 'best.pt'))
        torch.save({'model': model.state_dict()}, os.path.join(args.outdir, 'last.pt'))

    print(f"Best val Recall@1: {best_recall:.4f}")

if __name__ == "__main__":
    main()
