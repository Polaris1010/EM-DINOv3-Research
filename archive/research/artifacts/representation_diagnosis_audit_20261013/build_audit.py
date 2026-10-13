"""Regenerate evidence tables without modifying any historical input."""
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WS=ROOT.parents[1]


def read(relative):return json.loads((WS/relative).read_text(encoding="utf-8"))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,data):(ROOT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")


def main():
    control="artifacts/mitoverse_full_stage2a2_control_20261011"
    tail="artifacts/mitoverse_B_full_extend150k_20261012"
    labels="artifacts/mitoverse_CMCC_CMCT_label1_20261012"
    match=read(control+"/protocol_match.json")
    rows=[]
    for model,prefix,folder in [("P0","official",control),("B100k","100k",tail),("B150k","150k",tail)]:
        for dataset in ["CMCC","CMCT"]:
            path=f"{folder}/{prefix}_{dataset}_metrics.json"
            j=read(path)
            rows.append({"model":model,"dataset":dataset,"status":j["status"],"seed":j["seed"],"steps":j["steps"],
                         "decoder_initial_sha256":j.get("decoder_initial_sha256",match["decoder_initial_sha256"]),
                         "decoder_initial_sha256_evidence":"metrics" if "decoder_initial_sha256" in j else "protocol_match plus 100k reconstruction",
                         "encoder_sha256":j.get("encoder_sha256"),"encoder_path":j["encoder"],
                         "split_sha256":j["split_sha256"],"frozen_encoder":j.get("frozen_encoder",match["encoder_frozen"]),
                         "train_cases":j.get("train_cases"),"val_cases":j.get("val_cases"),
                         "macro_dice":next(iter(j["datasets"].values()))["macro_dice"],
                         "evidence_file":path,"evidence_file_sha256":digest(WS/path)})
    assert all(r["status"]=="complete" for r in rows)
    assert len({r["decoder_initial_sha256"] for r in rows})==1
    assert len({r["seed"] for r in rows})==1 and all(r["steps"]==3000 and r["frozen_encoder"] for r in rows)
    write("historical_probe_evidence.json",rows)
    with (ROOT/"historical_probe_evidence.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    source=WS/"artifacts/mitotopo_first_round/remote/source_snapshot/primus.py"
    assert digest(source)==match["dependency_snapshots"][0]["sha256"]
    official=(ROOT/"sources/official_primus.py").read_text(encoding="utf-8")
    assert "LaSt_FrequencyGate" not in official and "last_gates" not in official
    inputs=[WS/r["evidence_file"] for r in rows]+[source,
        WS/control/"protocol_match.json",WS/"artifacts/stage1_B_full.yaml",WS/"artifacts/stage1_train_B_full.py",
        WS/"artifacts/build_B_full_extend150k_20261012.py",WS/tail/"training_result_150k.json",
        WS/labels/"label_audit.json",WS/labels/"results.json",WS/labels/"frozen_probe_label1.py",
        WS/"artifacts/stage2a2_frozen_probe.py",WS/"artifacts/stage2a2_frozen_probe_control.py"]
    historical={str(p.relative_to(WS)).replace("\\","/"):digest(p) for p in inputs}
    target=ROOT/"historical_input_integrity.json"
    if target.exists():
        baseline=json.loads(target.read_text(encoding="utf-8"))
        assert baseline==historical,"Historical inputs changed"
    else:write("historical_input_integrity.json",historical)
    write("preservation_check.json",{"status":"PASS","files_checked":len(historical),"changed_files":[],
                                      "scope":"listed historical evidence inputs; no GPU/process operations"})
    expected=[1000,5000,10000,25000,50000,75000,100000,100010,105000,125000,150000]
    checkpoints=[{"step":s,"programmed_backbone_export":s in expected,
                  "status":"CONFIRMED_HISTORICAL_EXPORT" if s in [100000,150000] else "UNVERIFIED_LIVE_EXISTENCE",
                  "comment":"30k/60k are not milestones in formal B_full code" if s in [30000,60000] else ""}
                 for s in sorted(set(expected+[30000,60000]))]
    write("checkpoint_availability.json",checkpoints)
    exposure=[]
    labelaudit=read(labels+"/label_audit.json")
    mv=read("artifacts/data_governance_stage0_20261013/sources/mv_catalog.json")
    cropmap={"CMCC_crop113":"openorganelle_jrc_hela-2_recon-1_crop113",
             "CMCC_crop110":"openorganelle_jrc_macrophage-2_recon-1_crop110",
             "CMCC_crop3":"openorganelle_jrc_hela-2_recon-1_crop3",
             "CMCT_jrc_mus-liver_crop145":"openorganelle_jrc_mus-liver_recon-1_crop145",
             "CMCT_jrc_mus-kidney_crop156":"openorganelle_jrc_mus-kidney_recon-1_crop156"}
    for r in rows:
        for v in labelaudit["validation_labels"]:
            if v["dataset"]!="Dataset111_CMCC" and r["dataset"]=="CMCC":continue
            if v["dataset"]!="Dataset131_CMCT" and r["dataset"]=="CMCT":continue
            matched=cropmap[v["case"]]
            assert matched in mv
            exposure.append({"model":r["model"],"case":v["case"],"matched_mitoverse_volume":matched,
                             "ssl_training_pool_contains_corresponding_mv_crop":r["model"]!="P0",
                             "exact_preprocessed_pixel_or_roi_mapping":"UNVERIFIED",
                             "ssl_image_exposure":"corresponding MitoVerse source crop eligible/observed at volume level; exact ROI draw UNVERIFIED" if r["model"]!="P0" else "no project EM SSL; upstream raw-image overlap UNVERIFIED",
                             "supervised_exposure":"validation/checkpoint selection; not decoder gradient data",
                             "method_development_exposure":True,"annotation_complete":"UNVERIFIED: binary labels do not prove completeness",
                             "unseen_acquisition_claim_permitted":False})
    write("diagnosis_data_exposure_records.json",exposure)
    model_specs=[]
    for model in ["P0","B100k","B150k"]:
        evidence=next(r for r in rows if r["model"]==model)
        model_specs.append({"id":model,"checkpoint_path":evidence["encoder_path"],"sha256":evidence["encoder_sha256"] or "UNVERIFIED",
                            "resolved_config_path":match["dependency_snapshots"][2]["path"],
                            "resolved_config_sha256":match["dependency_snapshots"][2]["sha256"],
                            "dinov3_root":"/opt/data/3_117/tanghr/DINOv3/dinov3",
                            "nnunet_root":"/opt/data/3_117/tanghr/Med_DINOv3/MedDINOv3/nnUNet",
                            "primus_source_path":match["dependency_snapshots"][0]["path"],
                            "primus_source_sha256":match["dependency_snapshots"][0]["sha256"],
                            "ssl_manifest_sha256":"3fd614082bf584c21e212884856875e328104ea09a293a95510c0f5f3117edb9" if model!="P0" else None,
                            "readout_variant":"historical_primus_frequency_gated"})
    write("diagnosis_plan.template.json",{"status":"UNVERIFIED_ROIS_NOT_LOCKED","automatic_ssl_training":False,
        "models":model_specs,"decoder_seeds":[20260907,20261013,20261014],"factory_sha256":digest(ROOT/"tools/server_factory.py"),
        "rgb_mean":[.485,.456,.406],"rgb_std":[.229,.224,.225],"intensity_range":[0,255],
        "intensity_range_status":"EXAMPLE_ONLY: verify actual acquisition bit depth before locking",
        "rois":[],"roi_required_fields":["id","split (train or dev)","source_id","acquisition_id",
                    "label_kind (binary_mito or instance_mito)","image_path","label_path","valid_mask_path",
                    "image_sha256","label_sha256","valid_mask_sha256","annotation_evidence","spacing_yx_nm","roi_yx_nm","ssl_exposure"]})
    print(json.dumps({"historical_metrics_complete":len(rows),"historical_files_unchanged":len(historical),"live_server_audit":"UNVERIFIED"}))


if __name__=="__main__": main()
