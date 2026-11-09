"""HW3: MPS vs CPU timing for fine-tuning MiniLM. Prints only."""
import sys, time, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "training"))
import cdcp
from transformers import AutoTokenizer, AutoModel
texts = [e.text for e in cdcp.load("train")]
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
for dev in ["mps", "cpu"]:
    enc = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(dev)
    head = torch.nn.Sequential(torch.nn.Linear(384,128), torch.nn.ReLU(), torch.nn.Linear(128,4)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters())+list(head.parameters()), lr=2e-5)
    enc.train()
    def step(b):
        t = tok(b, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        h = enc(**t).last_hidden_state; m = t["attention_mask"].unsqueeze(-1).float()
        e = torch.nn.functional.normalize((h*m).sum(1)/m.sum(1), dim=1)
        loss = torch.nn.functional.cross_entropy(head(e), torch.zeros(len(b), dtype=torch.long, device=dev))
        opt.zero_grad(); loss.backward(); opt.step(); return loss.item()
    step(texts[:32]); step(texts[32:64])
    t0 = time.time(); n = 30
    for k in range(n): step(texts[64+32*k:96+32*k])
    tr = (time.time()-t0)/n
    enc.eval(); t0=time.time()
    with torch.no_grad():
        for k in range(0, 525, 128):
            t = tok(texts[k:k+128], padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev); enc(**t).last_hidden_state.sum().item()
    ev = time.time()-t0
    print(f"{dev}: train step {tr:.3f}s -> epoch (86 steps) {86*tr:.0f}s ; eval 525 train sents (dev-sized) {ev:.1f}s")
