"""Archive historical research code and Markdown, preserving bytes and provenance.

This reads only local research artifacts. It never runs archived experiments.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    "vendor", "deps", "__pycache__", "site-packages", "node_modules",
    "remote_snapshot", "research_bootstrap_20261013", "dinov3_reference",
    ".git", ".venv", "venv",
}
EXTENSIONS = {".md", ".py", ".sh", ".yaml", ".yml"}
SECRET_PATTERNS = [
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"gh[pousr]_[A-Za-z0-9]{30,}",
    r"github_pat_[A-Za-z0-9_]{30,}",
    r"hf_[A-Za-z0-9]{25,}",
    r"(?i)\b(?:password|passwd|secret|access_token|api_key)\s*=\s*([\"'])(?:(?!\1)[^\r\n]){6,}\1",
    r'(?i)["\x27](?:password|passwd|secret|access_token|api_key)["\x27]\s*:\s*(["\x27])(?:(?!\1)[^\r\n]){6,}\1',
]

def candidates(source_root: Path):
    artifacts = source_root / "artifacts"
    for directory, names, files in os.walk(artifacts):
        names[:] = sorted(n for n in names if n not in EXCLUDED and not n.startswith("upstream"))
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() in EXTENSIONS:
                yield path, path.relative_to(source_root)
    plan = source_root / "MitoVerse_全量重新预训练_新窗口执行方案.md"
    if plan.is_file():
        yield plan, Path(plan.name)

def archive(source_root: Path, check_only: bool = False):
    rows = []
    issues = []
    for source, relative in candidates(source_root):
        data = source.read_bytes()
        if len(data) >= 2_000_000:
            issues.append({"path": str(relative), "reason": "oversized text artifact"})
            continue
        text = data.decode("utf-8-sig")
        if any(re.search(pattern, text) for pattern in SECRET_PATTERNS):
            issues.append({"path": str(relative), "reason": "possible embedded credential"})
            continue
        target = (Path("docs/history") if source.suffix.lower() == ".md" else Path("archive/research")) / relative
        row = {
            "repository_path": target.as_posix(),
            "origin": str(source),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "provenance": "LOCAL_HISTORICAL_ARTIFACT",
            "execution_status": "UNVERIFIED_BY_ARCHIVAL_COPY",
        }
        if source.suffix.lower() == ".py":
            try:
                compile(text, str(relative), "exec")
                row["syntax_status"] = "VALID_FOR_LOCAL_PYTHON"
            except SyntaxError as error:
                row["syntax_status"] = "HISTORICAL_SYNTAX_ERROR"
                row["syntax_error_line"] = error.lineno
        rows.append((source, target, data, row))
    if issues:
        print(json.dumps({"status": "NEEDS_REVIEW", "issues": issues}, ensure_ascii=False))
        raise SystemExit(1)
    if check_only:
        print(json.dumps({"status": "READY", "files": len(rows), "bytes": sum(len(row[2]) for row in rows), "syntax_errors": [row[3] for row in rows if row[3].get("syntax_status") == "HISTORICAL_SYNTAX_ERROR"]}, ensure_ascii=False))
        return
    for source, target, data, row in rows:
        destination = ROOT / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != data:
            raise SystemExit(f"Refusing to overwrite an existing historical snapshot: {target}")
        destination.write_bytes(data)
    records = [row[3] for row in rows]
    index = ROOT / "evidence/history_file_index.json"
    index.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    markdown = [row[3] for row in rows if row[1].suffix.lower() == ".md"]
    lines = [
        "# 历次研究Markdown记录索引", "",
        "这里按原始目录保存本地已有研究记录，源码/配置对应档案位于 `archive/research/`。",
        "原文件字节不改动，每份的来源、大小、SHA见 `evidence/history_file_index.json`。",
        "档案中的计划、旧状态和旧watcher命令不代表当前完成事实或启动授权；当前事实仍以PROJECT_STATE、registry和NEXT_ACTION为准。",
        "", f"共保存 {len(markdown)} 份Markdown；全部历史代码/配置/记录 {len(records)} 份。", "",
        "| 原始记录 | 仓库副本 |", "| --- | --- |",
    ]
    for row in markdown:
        relative = Path(row["repository_path"]).relative_to("docs/history").as_posix()
        display = relative.replace("|", "\\|")
        lines.append(f"| {display} | [读取](<{relative}>) |")
    (ROOT / "docs/history/INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "ARCHIVED", "files": len(records), "markdown_files": len(markdown), "bytes": sum(row["bytes"] for row in records)}, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    archive(arguments.source_root.resolve(strict=True), arguments.check_only)
