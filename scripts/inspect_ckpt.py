import sys
import torch

p = sys.argv[1]
d = torch.load(p, map_location="cpu", weights_only=False)
print("top keys:", list(d.keys()))
pp = d.get("pipeline", d)
print("type pipeline:", type(pp))
if hasattr(pp, "keys"):
    keys = list(pp.keys())
    print(f"pipeline n={len(keys)}; first 30:")
    for k in keys[:30]:
        v = pp[k]
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")
    print("\nfields with 'gauss', 'means', 'scales', 'quats', 'opacit', 'features':")
    for k in keys:
        kl = k.lower()
        if any(s in kl for s in ("gauss", "means", "scales", "quats", "opacit", "features")):
            v = pp[k]
            print(f"  {k}: {tuple(v.shape) if hasattr(v,'shape') else type(v)}")
