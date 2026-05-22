from __future__ import annotations

import argparse, json, os, subprocess, sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: List[str], env=None):
    print("\n$ "+" ".join(map(str,cmd)), flush=True)
    subprocess.run(list(map(str,cmd)), cwd=ROOT, env=env, check=True)


def stage_name(mtokens:int)->str:
    if mtokens not in {1,5,20,100}:
        raise ValueError("--mtokens must be one of 1,5,20,100")
    return f"local_v3_{mtokens}m"


def paths(stage:str):
    processed=ROOT/"data"/"processed"
    out=ROOT/"outputs"
    train=processed/f"{stage}.train.jsonl"
    phase1=out/f"{stage}_phase1_v3_16k"
    ckpt=phase1/"last.pt"
    feat=processed/f"{stage}.memory_features.pt"
    mem=out/f"{stage}_memory_v3_16k"
    joint=out/f"{stage}_joint_v3_16k"
    return train, phase1, ckpt, feat, mem, joint


def filter_dialogue_blocks(src: Path, dst: Path, max_blocks:int=0):
    dst.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with src.open('r',encoding='utf-8') as f, dst.open('w',encoding='utf-8') as g:
        for line in f:
            if '"domain": "dialogue"' in line or '"domain":"dialogue"' in line:
                g.write(line); n+=1
                if max_blocks and n>=max_blocks: break
    if n==0:
        raise RuntimeError(f"no dialogue blocks found in {src}")
    return n


def main():
    ap=argparse.ArgumentParser(description="Local V3 16K Phase0→Phase3 runner with Chinese dialogue data and profiling.")
    ap.add_argument('--stage', choices=['all','phase0_data','audit_data','profile','phase1','generate','features','memory_v3','phase2_joint','phase3_sft','phase3_eval'], default='all')
    ap.add_argument('--mtokens', type=int, default=1, choices=[1,5,20,100])
    ap.add_argument('--seq_len', type=int, default=256)
    ap.add_argument('--phase1_steps', type=int, default=2000)
    ap.add_argument('--memory_v3_steps', type=int, default=500)
    ap.add_argument('--joint_steps', type=int, default=512)
    ap.add_argument('--phase3_steps', type=int, default=1000)
    ap.add_argument('--feature_samples', type=int, default=1000)
    ap.add_argument('--config', default='configs/local_v3_16k_phase1_4070.yaml')
    ap.add_argument('--safe_config', default='configs/local_v3_16k_phase1_4070_safe.yaml')
    ap.add_argument('--memory_config', default='configs/phase1_5b_v3_memory_4070_16k.yaml')
    ap.add_argument('--joint_config', default='configs/local_v3_16k_joint_4070.yaml')
    ap.add_argument('--sources', default='data/sources/local_v3_zh_dialogue_sources.yaml')
    ap.add_argument('--reuse_raw_cache', action='store_true')
    ap.add_argument('--hf_cache_dir', default='')
    args=ap.parse_args()
    env=dict(os.environ); env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(ROOT)+os.pathsep+env.get('PYTHONPATH',''); env.setdefault('PYTHONUNBUFFERED','1')
    st=stage_name(args.mtokens)
    train, phase1_dir, ckpt, feat, mem_dir, joint_dir = paths(st)
    docs=ROOT/'docs'; docs.mkdir(exist_ok=True)
    summary={'stage':st,'seq_len':args.seq_len,'sources':args.sources,'train':str(train),'phase1_ckpt':str(ckpt),'features':str(feat),'memory_dir':str(mem_dir)}
    (docs/'LOCAL_V3_PHASE0_TO_PHASE3_ACTIVE_PLAN.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

    if args.stage in {'all','phase0_data'}:
        cmd=[sys.executable,'scripts/data/run_data_pipeline.py','--stage',st,'--sources',args.sources,'--seq_len',args.seq_len,'--tokenizer_out','data/tokenizer/local_v3_32k','--output_dir','data/processed','--report_dir','data/reports','--raw_cache_dir','data/raw_cache','--intermediate_dir','data/intermediate','--tokenizer_vocab_size','32000']
        if args.reuse_raw_cache: cmd.append('--reuse_raw_cache')
        if args.hf_cache_dir: cmd += ['--hf_cache_dir', args.hf_cache_dir]
        run(cmd,env=env)
        if args.stage=='phase0_data': return
    if args.stage in {'all','audit_data'}:
        run([sys.executable,'scripts/data/audit_dataset.py','--train',f'data/processed/{st}.train.jsonl','--val',f'data/processed/{st}.val.jsonl','--test',f'data/processed/{st}.test.jsonl','--seq_len',args.seq_len,'--vocab_size','32000','--expected_mix','general=0.60,dialogue=0.20,reasoning=0.15,divergent=0.05','--out',f'data/reports/audit_{st}_v3.json'],env=env)
        if args.stage=='audit_data': return
    if args.stage in {'all','profile'}:
        run([sys.executable,'scripts/profile_local_v3.py','--config',args.safe_config,'--data',train,'--steps','10','--warmup','2','--out',f'docs/local_v3_profile_{st}.json'],env=env)
        if args.stage=='profile': return
    if args.stage in {'all','phase1'}:
        run([sys.executable,'scripts/train.py','--config',args.config,'--data',train,'--max_steps',args.phase1_steps,'--output_dir',phase1_dir],env=env)
        if args.stage=='phase1': return
    if args.stage in {'all','generate'}:
        run([sys.executable,'scripts/eval_generation_smoke.py','--config',args.config,'--checkpoint',ckpt,'--tokenizer','data/tokenizer/local_v3_32k/tokenizer.model','--out',f'docs/local_v3_generation_smoke_{st}.json'],env=env)
        if args.stage=='generate': return
    if args.stage in {'all','features'}:
        run([sys.executable,'scripts/extract_memory_features.py','--config',args.config,'--checkpoint',ckpt,'--data',train,'--seq_len',args.seq_len,'--max_samples',args.feature_samples,'--span_size','16','--batch_size','1','--out',feat,'--report',f'docs/LOCAL_V3_FEATURE_EXTRACTION_{st}.md'],env=env)
        if args.stage=='features': return
    if args.stage in {'all','memory_v3'}:
        run([sys.executable,'scripts/train_memory_v3.py','--config',args.memory_config,'--features',feat,'--checkpoint',ckpt,'--max_steps',args.memory_v3_steps,'--output_dir',mem_dir,'--report',f'docs/LOCAL_V3_MEMORY_REPORT_{st}.md'],env=env)
        if args.stage=='memory_v3': return
    if args.stage in {'all','phase2_joint'}:
        mem_ckpt=mem_dir/'memory_v3_last.pt'
        run([sys.executable,'scripts/train_joint_memory_lm.py','--config',args.joint_config,'--data',train,'--phase1_checkpoint',ckpt,'--memory_checkpoint',mem_ckpt,'--max_steps',args.joint_steps,'--output_dir',joint_dir,'--report',f'docs/LOCAL_V3_PHASE2_JOINT_REPORT_{st}.md'],env=env)
        if args.stage=='phase2_joint': return
    if args.stage in {'all','phase3_sft'}:
        dialogue=ROOT/'data'/'processed'/f'{st}.dialogue.train.jsonl'
        n=filter_dialogue_blocks(train, dialogue)
        phase3_dir=ROOT/'outputs'/f'{st}_phase3_dialogue_sft'
        run([sys.executable,'scripts/train.py','--config',args.config,'--data',dialogue,'--resume',ckpt,'--max_steps',args.phase3_steps,'--output_dir',phase3_dir],env=env)
        (docs/f'LOCAL_V3_PHASE3_DIALOGUE_SOURCE_{st}.json').write_text(json.dumps({'dialogue_blocks':n,'path':str(dialogue)},ensure_ascii=False,indent=2),encoding='utf-8')
        if args.stage=='phase3_sft': return
    if args.stage in {'all','phase3_eval'}:
        p3=ROOT/'outputs'/f'{st}_phase3_dialogue_sft'/'last.pt'
        if not p3.exists(): p3=ckpt
        run([sys.executable,'scripts/eval_generation_smoke.py','--config',args.config,'--checkpoint',p3,'--tokenizer','data/tokenizer/local_v3_32k/tokenizer.model','--out',f'docs/local_v3_phase3_generation_smoke_{st}.json'],env=env)
        if args.stage=='phase3_eval': return

if __name__=='__main__': main()
