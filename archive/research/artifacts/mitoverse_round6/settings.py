from pathlib import Path
ROOT=Path('/opt/data/3_117/tanghr')
BASE=Path(__file__).resolve().parent
R4=ROOT/'codex/mitoverse_round4_20260909_103239'
R5=ROOT/'codex/gram_long_and_explore_20260915_020459'
DATA=ROOT/'MitoVerse_data'
P1=ROOT/'DINOv3/med_checkpoints_continual_v2/eval/training_124999/teacher_checkpoint.pth'
P1_SHA='53f759c7019f8753d9d18b52aa79168b51b7480466934df47f46bc3d5f696524'
PYTHON=ROOT/'envs/meddinov3/bin/python'
