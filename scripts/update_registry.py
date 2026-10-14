"""Generate a YAML 1.2 (JSON syntax) registry from immutable evidence snapshots."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = "experiment_id status research_question dataset_manifest dataset_manifest_sha256 ssl_exposure supervised_train_split validation_split test_split backbone initial_checkpoint initial_checkpoint_sha256 final_checkpoint final_checkpoint_sha256 decoder decoder_source_commit decoder_checkpoint decoder_seed train_config resolved_config_sha256 code_commit python_executable global_batch optimizer_steps learning_rate tta normalization inference_protocol metrics_path report_path limitations".split()

def read(root: Path, name: str):
    return json.loads((root / name).read_text(encoding="utf-8"))

def sha(root: Path, name: str):
    return hashlib.sha256((root / name).read_bytes()).hexdigest()

def build_registry(root: Path = ROOT):
    models = {m["id"]: m for m in read(root, "experiments/MV-1/model_registry.json")["models"]}
    manifest = read(root, "configs/segmentation/MV-Dev-v1.json")
    protocol = read(root, "configs/evaluation/MV-1_protocol.yaml")
    hashes = read(root, "experiments/MV-2/input_hashes.json")
    inventory = {p["path"]: p for p in read(root, "evidence/checkpoint_inventory.json")}
    train = [e["id"] for e in manifest["entries"] if e["supervised_role"] == "train"]
    dev = [e["id"] for e in manifest["entries"] if e["supervised_role"] == "dev"]
    entries = []
    def entry(identifier, **fields):
        row = {k: "UNVERIFIED" for k in REQUIRED}
        row.update(experiment_id=identifier, limitations=[], **fields)
        entries.append(row)
        return row

    p0 = models["P0"]
    entry("P0-official", status="VERIFIED_COMPLETED", research_question="Official natural-image initialization reference", backbone="DINOv3 ViT-L/16", final_checkpoint=p0["checkpoint_path"], final_checkpoint_sha256=p0["sha256"], code_commit="91ffb6803c74aeb6927209b82560142743763dd7", evidence=["experiments/MV-1/model_registry.json", "evidence/checkpoint_inventory.json"], hash_verification="Recorded hash, safely loaded CPU state dict; hash not recomputed this takeover")
    for model, step in [("B100k", 100000), ("B150k", 150000)]:
        folder = f"experiments/B_full/{model}"
        result = read(root, folder + "/result.json")
        checkpoint = read(root, folder + "/latest_checkpoint.json")
        m = models[model]
        initial = p0 if model == "B100k" else read(root, "experiments/B_full/B100k/latest_checkpoint.json")
        row = entry(model, status="VERIFIED_COMPLETED" if result["status"] == "complete" and result["completed_steps"] == step else "UNVERIFIED", research_question="Full MitoVerse SSL development representation" if model == "B100k" else "Additional 50k continuation at original minimum LR", dataset_manifest="configs/pretrain/pretrain_all_mitoverse.json", dataset_manifest_sha256=m["ssl_manifest_sha256"], ssl_exposure="All 212 MitoVerse volumes, including historical train/validation/test/excluded images; labels not consumed by SSL loader", backbone="DINOv3 ViT-L/16", initial_checkpoint=initial.get("checkpoint_path", initial.get("path")), initial_checkpoint_sha256=initial["sha256"], final_checkpoint=m["checkpoint_path"], final_checkpoint_sha256=m["sha256"], train_config="configs/pretrain/B_full.yaml", resolved_config_sha256=sha(root, "configs/pretrain/B_full_resolved.yaml"), code_commit=result["dinov3_commit"], python_executable="/opt/data/3_117/tanghr/dinov3_shared/bin/python", global_batch=result["effective_global_batch"], optimizer_steps=result["completed_steps"], learning_rate={"effective_peak_lr": result["effective_peak_lr"], "warmup_steps": result["warmup_steps"], "schedule": result.get("schedule", "Recorded 100k official cosine schedule")}, normalization="configs/pretrain/B_full.yaml crops.rgb_mean/rgb_std", metrics_path=folder + "/result.json", report_path=folder + ("/training_report.md" if model == "B100k" else "/REPORT_100k_150k_zh.md"), evidence=[folder + "/result.json", folder + "/checkpoint_metadata.json", "evidence/checkpoint_inventory.json"], full_optimizer_checkpoint=checkpoint["path"], full_optimizer_checkpoint_sha256=checkpoint["sha256"], teacher_checkpoint=m["checkpoint_path"], student_checkpoint=m["checkpoint_path"].replace("teacher_backbone", "student_backbone"), trainer_sha256=result["trainer_sha256"], sampler_sha256=result["sampler_sha256"], gram="OFF", hra="NOT_RUN", last="OFF", topology="OFF", limitation_notes=["Transductive development asset; no unseen-image claim", "Large checkpoint hashes are existing server records, not rehashed", "Safe full optimizer deserialization unavailable under weights_only=True; sidecar metadata and teacher/student exports verified"])
        row["limitations"] = row.pop("limitation_notes")
        row["student_metadata"] = inventory.get(row["student_checkpoint"], "UNVERIFIED")
        if model == "B150k":
            row["additional_optimizer_steps"] = 50000
            row["resolved_config_note"] = "Base config is identical to B100k; runtime continuation overrides in result.json are authoritative. Not an independent 150k-from-zero schedule."
            row["report_readout_scope"] = "Historical modified/gated Primus 100k/150k downstream comparison; original Primus results are separately registered under CMCC-CMCT-original entries."

    for model in ["P0", "B100k", "B150k"]:
        for seed in protocol["seed_sequence"]:
            folder = f"experiments/MV-1/runs/{model}/seed{seed}"
            run = read(root, folder + "/run_protocol.json")
            metrics = read(root, folder + "/metrics.json")
            m = models[model]
            checkpoint = str(Path(m["primus_source_path"]).parent.parent).replace("\\", "/") + f"/runs/{model}/seed{seed}/head_last.pth"
            metadata = inventory[checkpoint]
            row = entry(f"MV-1-{model}-seed{seed}", status="VERIFIED_COMPLETED" if metrics["status"] == "complete" and metadata["metadata"]["step"] == protocol["locked_steps"] else "UNVERIFIED", research_question="Matched frozen original Primus development readout", dataset_manifest="configs/segmentation/MV-Dev-v1.json", dataset_manifest_sha256=run["manifest_sha256"], ssl_exposure="Official natural-image pretraining" if model == "P0" else "transductive_full_mitoverse", supervised_train_split=train, validation_split=dev, test_split="NOT_ACCESSED", backbone="DINOv3 ViT-L/16 frozen", initial_checkpoint=m["checkpoint_path"], initial_checkpoint_sha256=run["encoder_sha256"], final_checkpoint=checkpoint, final_checkpoint_sha256=hashes[checkpoint], decoder="Original MedDINOv3 Primus_Multiscale", decoder_source_commit="a2460c2547899283446e520b84523454f7e25608", decoder_checkpoint=checkpoint, decoder_seed=seed, train_config="configs/evaluation/MV-1_protocol.yaml", resolved_config_sha256=run["protocol_sha256"], code_commit="UNVERIFIED", python_executable="/opt/data/3_117/tanghr/envs/meddinov3/bin/python", global_batch=run["training_batch_size"], optimizer_steps=metrics["completed_steps"], learning_rate=protocol["lr"], tta=False, normalization=protocol["input_normalization"], inference_protocol={"window":256,"stride":128,"tile_batch_size":run["validation_tile_batch_size"],"aggregation":"mean softmax probabilities","prediction":"argmax; no postprocessing","selection":"fixed 20000 steps; not per-model best validation"}, metrics_path=folder + "/per_volume_metrics.csv", report_path="experiments/MV-1/source_failure_analysis.md", evidence=[folder + "/run_protocol.json", folder + "/metrics.json", "experiments/MV-1/checkpoint_inventory.json", "experiments/MV-2/input_hashes.json"], probe_source_sha256=run["probe_source_sha256"], decoder_source_sha256=m["primus_source_sha256"], decoder_initial_state_sha256=run["decoder_initial_sha256"], encoder_frozen=run["encoder_frozen"], frequency_gates=run["gates"], parameter_count=metadata["head_parameters_in_state"], python_evidence="Completed launcher source plus actual metric logs naming meddinov3 site-packages", hash_verification="Existing MV-2 input_hashes record plus current safe CPU checkpoint metadata; hash not recomputed")
            row["limitations"] = ["4 development volumes across 3 sources; decoder seeds do not multiply biological acquisitions", "Source/acquisition independence remains UNVERIFIED", "B100/B150 results are transductive development evidence"]

    cm = read(root, "experiments/CMCC_CMCT/results.json")
    for result in cm["summary"]:
        model = result["model"]
        row = entry(f"CMCC-CMCT-original-{result['dataset']}-{model}", status="VERIFIED_COMPLETED" if cm["status"] == "complete" else "UNVERIFIED", research_question="Original versus historical gated Primus reference comparison", backbone="DINOv3 ViT-L/16 frozen", initial_checkpoint=models[model]["checkpoint_path"], initial_checkpoint_sha256=models[model]["sha256"], decoder=cm["decoder"], decoder_source_commit=cm["upstream_commit"], decoder_seed=20260907, python_executable="/opt/data/3_117/tanghr/envs/meddinov3/bin/python", optimizer_steps=3000, tta="four_flip", metrics_path="experiments/CMCC_CMCT/results.json", report_path="experiments/CMCC_CMCT/REPORT_zh.md", evidence=["experiments/CMCC_CMCT/results.json", "evidence/primus/github_source_verification.json"], selected_step=result["selected_step"], dice=result["dice_mean"], decoder_source_sha256=cm["decoder_sha256"], source_registry_conflict="The historical model_registry.json in this folder still describes the gated factory. Completed results and the separately verified original source govern these six records.")
        row["limitations"] = ["Historical four-flip TTA; not MV-1/MV-2 inference protocol", "Surface metrics voxel-equivalent HD95 and NSD@1; do not relabel as nm", "Decoder checkpoints and complete resolved inference config remain UNVERIFIED in this takeover index"]

    mv2 = read(root, "configs/evaluation/MV-2_manifest.json")
    status = read(root, "experiments/MV-2/STATUS.json")
    row = entry("MV-2-notta", status="VERIFIED_RUNNING" if status["status"] == "running" else "UNVERIFIED", research_question="Does B100k improve over P0 across expanded development sources?", dataset_manifest="configs/evaluation/MV-2_manifest.json", dataset_manifest_sha256=sha(root, "configs/evaluation/MV-2_manifest.json"), ssl_exposure="transductive_full_mitoverse", supervised_train_split=mv2["head_supervised_train_ids"], validation_split=[e["id"] for e in mv2["entries"]], test_split={"reserved_ids":mv2["reserved_test_ids"],"policy":"excluded from development evaluation"}, backbone="DINOv3 ViT-L/16 frozen", decoder="Original MedDINOv3 Primus_Multiscale", decoder_source_commit="a2460c2547899283446e520b84523454f7e25608", decoder_checkpoint=[e["decoder_checkpoint"] for e in entries if e["experiment_id"].startswith("MV-1-")], decoder_seed=protocol["seed_sequence"], python_executable="/opt/data/3_117/tanghr/envs/meddinov3/bin/python", optimizer_steps=0, tta=False, normalization=mv2["normalization"], inference_protocol={k:mv2[k] for k in ['inference_function','threshold','crop_size','stride','tile_batch_size','reuse_existing_MV1_dev_results']}, metrics_path="UNVERIFIED", report_path="UNVERIFIED", evidence=["experiments/MV-2/STATUS.json", "evidence/server_runtime.json", "configs/evaluation/MV-2_manifest.json"], completion_verified=False, inference_only=True, existing_job=True, server_root="/opt/data/3_117/tanghr/codex/mv_dev_v2_notta_20261013", active_service="mv-dev-v2-notta-20261013.service", inherited_experiments=[e["experiment_id"] for e in entries if e["experiment_id"].startswith("MV-1-")], volume_roi_count=len(mv2["entries"]), source_count=len({e["source_id"] for e in mv2["entries"]}))
    row["limitations"] = mv2["limitations"] + ["Snapshot read during a live job; no final expanded metric claim", "Mixed published validation and exploratory custom development cohorts; report separately", "188 volumes/ROIs do not equal 188 independent acquisitions", "Future updates must retain this dated snapshot"]
    entry("MV-2-four-flip-superseded", status="VERIFIED_FAILED", research_question="Historical protocol error; results excluded from official MV-2", tta="four_flip", execution_state="SUPERSEDED", evidence=["experiments/MV-2/superseded_tta_status.json"], report_path="experiments/MV-2/superseded_tta_status.json", scientific_failure=False, failure_type="protocol superseded by user correction")
    gate = read(root, "experiments/history/stage2A_gate.json")
    entry("Stage2A", status="VERIFIED_FAILED" if gate["stage2A_gate"] == "FAIL" else "UNVERIFIED", research_question="Topology crop-position calibration", metrics_path="experiments/history/stage2A_gate.json", report_path="experiments/history/stage2A_gate.json", evidence=["experiments/history/stage2A_gate.json"], auto_launch_T1=gate["auto_launch_T1"], T1_status=gate["T1_status"])
    entry("Stage2A2-P", status="PAUSED", research_question="User-paused topology branch", evidence=["experiments/history/stage2A2_STATUS.json"])
    entry("Stage2A2-E", status="VERIFIED_COMPLETED", research_question="Historical frozen probe infrastructure", report_path="experiments/history/stage2A2_STATUS.json", evidence=["experiments/history/stage2A2_STATUS.json"])
    entry("Stage2A2-S", status="VERIFIED_COMPLETED", research_question="Completed negative crop pilot / offline GT diagnostic", report_path="experiments/history/stage2A2_STATUS.json", evidence=["experiments/history/stage2A2_STATUS.json"], long_training_admitted=False, final_human_gate="UNVERIFIED")
    entry("B_clean", status="PLANNED", research_question="Future image-isolated inductive SSL reference; not a dependency of current development evaluation", execution_state="NOT_STARTED", evidence=["docs/DATA_PROTOCOL.md"], authorized_to_launch=False)
    entry("legacy-31-volume-B-L-T-LT", status="UNVERIFIED", research_question="Superseded historical 31-volume experiments", server_root="/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227", execution_state="HISTORY_ONLY; do not restore watchers", evidence=["docs/PROJECT_STATE.md"])
    index_path=root/'experiments/MV-3/run_index.json'
    if index_path.exists():
        index=read(root,'experiments/MV-3/run_index.json')
        mv3=read(root,index['manifest']);config=read(root,index['protocol'])
        for run in index['runs']:
            model,seed=run['model'],run['seed'];encoder=models[config['models'][model]]
            row=entry(f'MV-3-{model}-seed{seed}',status=run['status'],research_question='Does full multisource supervision preserve the B100k advantage over P0?',dataset_manifest=index['manifest'],dataset_manifest_sha256=sha(root,index['manifest']),ssl_exposure='Official natural image pretraining' if model=='G-P0' else 'transductive_full_mitoverse',supervised_train_split=[r['id'] for r in mv3['entries'] if r['supervised_role']=='train'],validation_split=[r['id'] for r in mv3['entries'] if r['supervised_role']=='dev'],test_split={'reserved_ids':mv3['reserved_test_ids'],'policy':'never read for supervision or model selection'},backbone='DINOv3 ViT-L/16 frozen',initial_checkpoint=encoder['checkpoint_path'],initial_checkpoint_sha256=encoder['sha256'],final_checkpoint=run.get('checkpoint_path','UNVERIFIED'),final_checkpoint_sha256=run.get('checkpoint_sha256','UNVERIFIED'),decoder='Original MedDINOv3 Primus_Multiscale',decoder_source_commit='a2460c2547899283446e520b84523454f7e25608',decoder_checkpoint=run.get('checkpoint_path','UNVERIFIED'),decoder_seed=seed,train_config=index['protocol'],resolved_config_sha256=sha(root,index['protocol']),code_commit=index.get('code_commit','UNVERIFIED'),python_executable='/opt/data/3_117/tanghr/envs/meddinov3/bin/python',global_batch=1,optimizer_steps=run.get('completed_steps','UNVERIFIED'),learning_rate=config['lr'],tta=False,normalization=config['normalization'],inference_protocol={'selection':config['selection'],'final_validation':config['final_validation'],'tile_batch_size':16},metrics_path=run.get('metrics_path','UNVERIFIED'),report_path=run.get('report_path','docs/MV3_PROTOCOL.md'),evidence=['experiments/MV-3/run_index.json',index['manifest'],index['protocol']],server_root=index['server_root'],planned_optimizer_steps=40000,encoder_frozen=True,frequency_gates=False,train_source_count=12,train_regions=156,dev_regions=38,execution_state=run.get('execution_state',run['status']),decoder_initial_sha256=run.get('decoder_initial_sha256','UNVERIFIED'),sampling_hash=run.get('sampling_hash','UNVERIFIED'))
            row['limitations']=mv3['limitations']
    return {"schema_version":"1.0", "serialization":"JSON syntax, valid YAML 1.2; stdlib reader", "observed_at":read(root,"evidence/collection_receipt.json")["assets_collected_at"], "unknown_value":"UNVERIFIED", "statuses":["VERIFIED_COMPLETED","VERIFIED_FAILED","PAUSED","PLANNED","UNVERIFIED","VERIFIED_RUNNING"], "checkpoint_hash_policy":"Use existing recorded server hashes; safe CPU metadata joins verified independently. No full checkpoint rehash in takeover.", "experiments":entries}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true",help="Verify committed registry equals generated evidence without writing")
    args=parser.parse_args()
    expected=json.dumps(build_registry(),ensure_ascii=False,indent=2)+"\n"
    path=ROOT/"experiments/registry.yaml"
    if args.check:
        if path.read_text(encoding="utf-8")!=expected:raise SystemExit("Registry is stale; review evidence, run scripts/update_registry.py, commit the change.")
        print("Registry generation matches evidence")
    else:
        path.write_text(expected,encoding="utf-8",newline="\n")
        print(f"Generated {len(build_registry()['experiments'])} experiment records")

if __name__=="__main__":main()
