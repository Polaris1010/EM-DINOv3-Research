"""Execute audited decoder classes on CPU, without importing a server environment.

AST extraction preserves class bodies. Framework-only base/initialization are
provided locally. This verifies decoder code, NOT real DINOv3 checkpoint outputs.
LayerNormNd body is taken from the captured primary upstream source. The server's
installed LayerNormNd version remains UNVERIFIED until a live inventory is read.
"""
import ast
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
import numpy as np
import torch
from torch import nn
from diagnosis import FrozenReadout, PrimusHead, LinearProbe, masked_loss, digest, sha

WORKSPACE = ROOT.parents[1]
HISTORICAL = WORKSPACE / "artifacts/mitotopo_first_round/remote/source_snapshot/primus.py"


def classes_from(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in body} != set(names):
        raise ValueError("Missing audited source classes")
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),"exec"),namespace)


def load_primus_source(path):
    env = {"torch":torch,"nn":nn,"np":np,"math":math,"AbstractDynamicNetworkArchitectures":nn.Module}
    classes_from(ROOT/"sources/reference_patch_encode_decode.py",["LayerNormNd"],env)
    class InitWeights_He:
        def __init__(self, negative_slope): self.negative_slope = negative_slope
        def __call__(self, module):
            if isinstance(module,(nn.Conv2d,nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight,a=self.negative_slope)
                if module.bias is not None: nn.init.zeros_(module.bias)
    env["InitWeights_He"] = InitWeights_He
    names = ["PatchDecode","Decoder","Primus_Multiscale"]
    if "class LaSt_FrequencyGate" in path.read_text(encoding="utf-8"):
        names.insert(0,"LaSt_FrequencyGate")
    classes_from(path,names,env)
    return env["Primus_Multiscale"]


class SyntheticEncoder(nn.Module):
    """Explicit fake encoder; never a substitute for natural/B_full weights."""
    def __init__(self, embed_dim):
        super().__init__()
        self.proj=nn.Conv2d(3,embed_dim,16,stride=16)

    def get_intermediate_layers(self,x,n,reshape=True):
        f=self.proj(x)
        return [f*(i+1)/4 for i in range(len(n))]


def run():
    torch.set_num_threads(2)
    cls=load_primus_source(HISTORICAL)
    official_cls=load_primus_source(ROOT/"sources/official_primus.py")
    with torch.device("meta"):
        m=cls(1024,16,2,dino_encoder=None,interaction_indices=[5,11,17,23])
        official=official_cls(1024,16,2,dino_encoder=None,interaction_indices=[5,11,17,23])
        counts={"up_projection":sum(p.numel() for p in m.up_projection.parameters()),
                "last_gates":sum(p.numel() for p in m.last_gates.parameters()),
                "total_head":sum(p.numel() for p in m.parameters()),
                "official_ungated_head":sum(p.numel() for p in official.parameters())}
        shapes=[{"name":k,"shape":list(p.shape),"numel":p.numel(),"trainable_by_default":p.requires_grad}
                for k,p in m.named_parameters()]
    # Actual historical head dimensions, small spatial footprint; fake encoder.
    torch.manual_seed(20260907)
    encoder=SyntheticEncoder(1024)
    torch.manual_seed(20260907)
    primus=cls(1024,16,2,dino_encoder=encoder,interaction_indices=[5,11,17,23])
    net=FrozenReadout(encoder,PrimusHead(primus))
    before=digest(encoder)
    head_init=digest(net.head)
    x=torch.rand(1,1,32,32)
    target=torch.zeros(1,32,32,dtype=torch.long);target[:,8:24,8:24]=1
    valid=torch.ones_like(target,dtype=torch.bool);valid[:,:2]=False
    net.train()
    y=net(x)
    loss=masked_loss(y,target,valid)
    loss.backward()
    grad_modules={name:all(p.grad is not None and torch.isfinite(p.grad).all() for p in mod.parameters())
                  for name,mod in [("up_projection",net.head.up_projection),("last_gates",net.head.last_gates)]}
    assert y.shape==(1,2,32,32) and torch.isfinite(loss)
    assert before==digest(encoder) and all(p.grad is None for p in encoder.parameters())
    assert all(grad_modules.values()) and not encoder.training
    counts.update({"evidence":"CPU meta construction of exact captured Primus class bodies with upstream LayerNormNd",
                   "encoder_parameter_count":"UNVERIFIED: no real checkpoint instantiated",
                   "server_installed_layernorm_version":"UNVERIFIED",
                   "historical_source_sha256":sha(HISTORICAL),"parameters":shapes})
    smoke={"status":"PASS","torch":torch.__version__,"device":"cpu","real_encoder_weights":False,
           "encoder":"SyntheticEncoder","primus_embed_dim":1024,"input_size":32,
           "loss":float(loss.detach()),"head_init_sha256_local_synthetic_context":head_init,
           "encoder_unchanged":True,"encoder_grads_absent":True,"encoder_eval_in_train":True,
           "decoder_gradient_finite":grad_modules,"framework_imports":"CPU AST extraction; not full server dependency environment"}
    (ROOT/"model_parameter_summary.json").write_text(json.dumps(counts,indent=2),encoding="utf-8")
    (ROOT/"decoder_cpu_smoke.json").write_text(json.dumps(smoke,indent=2),encoding="utf-8")
    print(json.dumps({"smoke":smoke,"parameter_counts":{k:counts[k] for k in list(counts)[:4]}},indent=2))


if __name__=="__main__": run()
