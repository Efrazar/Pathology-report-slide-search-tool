"""Step 11 (Track 2): turn every PanNuke tile into one vector with H-optimus-0.

An "embedding" is the model's summary of a tile: 1,536 numbers. Tiles that look alike to the
model get vectors that point in similar directions. Steps 12 and 13 only ever see these
vectors, never the pixels, so this is the only step that needs the GPU. Run it once.

Before the first run:
- accept the model's terms at https://huggingface.co/bioptimus/H-optimus-0
- run `hf auth login` (older huggingface_hub: `huggingface-cli login`) with a read token (huggingface.co > Settings > Access Tokens)
- install the slide extras: pip install -e ".[slides]"
- check: python -c "import torch; print(torch.cuda.is_available())" should print True

Choices, and why:
- fp16 through torch.autocast. The RTX 2080 Ti (Turing) has fast fp16 but no bf16. Autocast
  runs the big matrix multiplications in fp16 and keeps the touchy parts (e.g. LayerNorm)
  in fp32, which is safer than converting the whole model with .half().
- Tiles are resized from 256 to 224 pixels, the size the model was trained on. Caveat:
  H-optimus-0 was trained at about 0.5 micrometre per pixel (20x) and PanNuke is mostly 40x,
  so the tiles look more zoomed in than the model is used to. We keep this simple first,
  and can test a 2x downscale later if the scores call for it.
- The normalisation numbers come from the model card. They must match training, or the model
  sees colours it never saw.

TCGA hospital study (after script 21):  python scripts/11_embed_tiles.py --dataset tcga

Output: data/interim/embeddings/<model>_fold{1,2,3}.npy (or <model>_tcga.npy with
--dataset tcga), float32, N x 1536, and a copy of the
labels next to each, so steps 12 and 13 do not need the 1.5 GB of tiles at all.

Smoke test (no GPU, no download, random weights; checks that the pipeline runs):
    python scripts/11_embed_tiles.py --smoke
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import timm

from prss.slides.data import FOLDS, PANNUKE, ROOT, load_fold, load_tcga_tiles

MODELS = {
    # name: (timm id, extra create_model arguments, mean, std)
    "hoptimus0": ("hf-hub:bioptimus/H-optimus-0", dict(init_values=1e-5, dynamic_img_size=False),
                  (0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)),
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="hoptimus0", choices=list(MODELS))
ap.add_argument("--batch", type=int, default=32, help="lower it if you get CUDA out of memory")
ap.add_argument("--data", default=str(PANNUKE))
ap.add_argument("--dataset", default="pannuke", choices=["pannuke", "tcga"],
                help="tcga: the hospital-study tiles from script 21")
ap.add_argument("--smoke", action="store_true", help="tiny random model, first 64 tiles per part")
ap.add_argument("--limit", type=int, help="embed only the first N tiles per part (default: all; 64 with --smoke)")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
if args.smoke:
    name, model = "smoke", timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
else:
    if device == "cpu":
        raise SystemExit("No CUDA GPU found. This model would take hours on the CPU; use --smoke to test.")
    name = args.model
    hub_id, kwargs, mean, std = MODELS[name]
    model = timm.create_model(hub_id, pretrained=True, **kwargs)
model = model.eval().to(device)
print(f"model {name} on {device}" + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""))
mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
std = torch.tensor(std, device=device).view(1, 3, 1, 1)

out_dir = ROOT / "data" / "interim" / "embeddings"
out_dir.mkdir(parents=True, exist_ok=True)

# Each "part" is embedded and saved separately: the 3 PanNuke folds, or the one TCGA tile set.
# For TCGA the saved "types" are the hospital (tissue source site) codes.
if args.dataset == "pannuke":
    parts = [(f"fold{k}", lambda k=k: load_fold(k, args.data)) for k in FOLDS]
else:
    parts = [("tcga", load_tcga_tiles)]

for k, load in parts:
    images, types = load()
    n = min(args.limit or (64 if args.smoke else len(images)), len(images))
    vecs, t0 = [], time.time()
    for i in range(0, n, args.batch):
        # uint8 (B, 256, 256, 3) -> float (B, 3, 224, 224), scaled to 0-1, then normalised
        # np.array copies the batch off the read-only memory map, so torch gets a normal array
        x = torch.from_numpy(np.array(images[i:min(i + args.batch, n)])).to(device)
        x = x.permute(0, 3, 1, 2).float().div(255)
        x = F.interpolate(x, size=224, mode="bilinear", antialias=True, align_corners=False)
        x = (x - mean) / std
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16,
                                                     enabled=device == "cuda"):
            vecs.append(model(x).float().cpu().numpy())
        if (i // args.batch) % 20 == 0:
            done = min(i + args.batch, n)
            print(f"  {k}: {done}/{n} tiles, {done / (time.time() - t0):.0f} tiles/s", flush=True)
    emb = np.concatenate(vecs)
    if not np.isfinite(emb).all():   # fp16 overflow shows up as inf/nan; better to stop than to score it
        raise SystemExit(f"{k}: non-finite values in the embeddings")
    np.save(out_dir / f"{name}_{k}.npy", emb)
    np.save(out_dir / f"{name}_{k}_types.npy", types[:n])
    print(f"{k}: saved {emb.shape} in {time.time() - t0:.0f} s")
    if device == "cuda":
        print(f"  peak GPU memory so far: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
