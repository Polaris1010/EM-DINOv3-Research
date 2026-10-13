"""Bounded local verification, CPU only; captures exact outputs in the delivery."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    results=[]
    for relative in ["tools/diagnosis.py","tools/source_smoke.py","tools/server_factory.py","tools/collect_live_evidence.py",
                     "build_audit.py","tests/test_diagnosis.py"]:
        path=ROOT/relative
        ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    commands=[[sys.executable,str(ROOT/"build_audit.py")],
              [sys.executable,str(ROOT/"tools/source_smoke.py")],
              [sys.executable,"-m","unittest","discover","-s",str(ROOT/"tests"),"-v"],
              [sys.executable,str(ROOT/"tools/diagnosis.py"),"--plan",str(ROOT/"diagnosis_plan.template.json")]]
    for i,command in enumerate(commands):
        proc=subprocess.run(command,capture_output=True,text=True,encoding="utf-8",errors="replace")
        expected_rejection=i==3
        passed=proc.returncode!=0 and "locked development plan" in proc.stderr if expected_rejection else proc.returncode==0
        results.append({"command":command,"exit_code":proc.returncode,"expected_template_rejection":expected_rejection,
                        "passed":passed,"stdout":proc.stdout,"stderr":proc.stderr})
    report={"status":"PASS" if all(r["passed"] for r in results) else "FAIL","cpu_only":True,
            "real_dino_checkpoints_tested":False,"unit_tests":8,"commands":results}
    (ROOT/"verification_results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    hashes={}
    for path in ROOT.rglob("*"):
        if path.is_file() and "vendor" not in path.parts and "__pycache__" not in path.parts and path.name!="delivery_sha256.json":
            hashes[str(path.relative_to(ROOT)).replace("\\","/")]=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/"delivery_sha256.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    print(json.dumps({"status":report["status"],"unit_tests":8,"real_dino_checkpoints_tested":False,"delivery_files":len(hashes)}))
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":sys.exit(main())
