"""Derive the matched probe by replacing only the decoder import/forward."""
from pathlib import Path
import hashlib,ast,json
R=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def replace(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
def main():
    reference=R/'src/probe_label1_reference.py'
    s=reference.read_text(encoding='utf-8')
    official_hash=sha(R/'src/official_primus.py')
    assert 'last_gates' not in (R/'src/official_primus.py').read_text()
    loader="""    import importlib.util,inspect
    official_path=RUN/'src/official_primus.py'
    assert digest(official_path)=='OFFICIAL_HASH'
    spec=importlib.util.spec_from_file_location('meddinov3_official_primus',official_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    Primus_Multiscale=module.Primus_Multiscale
    assert Path(inspect.getfile(Primus_Multiscale)).resolve()==official_path.resolve()""".replace('OFFICIAL_HASH',official_hash)
    s=replace(s,'    from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import Primus_Multiscale',loader)
    s=replace(s,'    net.cuda();del ck,sd;gc.collect();return net',"    assert not hasattr(net,'last_gates') and not any('gate' in n.lower() for n,_ in net.named_parameters())\n    assert sum(p.numel() for p in net.parameters() if p.requires_grad)==44576514\n    net.cuda();del ck,sd;gc.collect();return net")
    s=replace(s,'                logits=net.up_projection(torch.cat([gate(f.float()) for gate,f in zip(net.last_gates,features)],dim=1))',
                  '                logits=net.up_projection(torch.cat([f.float() for f in features],dim=1))')
    s=s.replace('fresh Primus decoder and gates','fresh official MedDINOv3 Primus decoder without gates')
    s=replace(s,"\n    (out/'protocol.json').write_text(json.dumps(result,indent=2))", "\n    result.update({'decoder_variant':'official_meddinov3_primus_multiscale','decoder_source':str(RUN/'src/official_primus.py'),'decoder_source_sha256':digest(RUN/'src/official_primus.py'),'decoder_upstream_commit':'a2460c2547899283446e520b84523454f7e25608','frequency_gates':False})\n    (out/'protocol.json').write_text(json.dumps(result,indent=2))")
    # Above protocol write occurs once at this indentation (the loop has 8 spaces).
    compile(s,str(R/'src/frozen_probe_official.py'),'exec')
    (R/'src/frozen_probe_official.py').write_text(s,encoding='utf-8')
    def funcs(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a=funcs(reference.read_text(encoding='utf-8'));b=funcs(s)
    assert all(a[k]==b[k] for k in ['seed','digest','head_digest','Samples','loss','validation'])
    (R/'build_audit.json').write_text(json.dumps({'reference_sha256':sha(reference),'official_primus_sha256':official_hash,
        'probe_sha256':sha(R/'src/frozen_probe_official.py'),'matched_functions':['seed','digest','head_digest','Samples','loss','validation'],
        'changes':['official source import','remove gates from training forward','decoder provenance metadata','absence/count assertions']},indent=2))
    print('Built official probe',official_hash)
if __name__=='__main__':main()
