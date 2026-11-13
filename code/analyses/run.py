"""Plan and execute a named study locally, without AWS/SLURM imports or services."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO.parent))
from NM_TinyRNN.code.analyses import get_study,common,execution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','task','collect'])
    parser.add_argument('--study',choices=['baseline','pooled_ws','data_scaling'],default='baseline')
    parser.add_argument('--config',type=Path,help='JSON overrides for planning; omitted uses study defaults')
    parser.add_argument('--data-root',type=Path,default=REPO/'data/AB_behaviour')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args(argv)
    root = args.output.resolve()
    if args.action=='plan':
        config = json.loads(args.config.read_text()) if args.config else {}
        config.setdefault('name',args.study)
        study = get_study(args.study)
        try:
            plan = study.build_plan(config,args.data_root.resolve())
        except (ValueError,NotImplementedError) as error:
            parser.error(str(error))
        hashes = {}
        source_files = [REPO/path for path in dict.fromkeys(common.SOURCE_FILES+study.SOURCE_FILES)]
        paths = source_files+[p for files in plan['data_files'].values() for p in files]
        for path in paths:
            hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        identity = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
        manifest = dict(identity=identity,files=hashes,data_root=str(args.data_root.resolve()))
        if (root/'local-inputs.json').exists():
            raise FileExistsError('A local plan already exists; use a new output directory')
        common.write_json(root/'run-config.json',plan['config'])
        common.write_json(root/'plan.json',common.plan_payload(plan))
        common.write_json(root/'local-inputs.json',manifest)
        print(json.dumps(dict(study=plan['config']['study'],experiments=len(plan['experiment_df']),
                              tasks=len(plan['task_df']),output=str(root))))
        return
    manifest = json.loads((root/'local-inputs.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Local input/source changed: {name}; create a new plan')
    config = json.loads((root/'run-config.json').read_text())
    payload = json.loads((root/'plan.json').read_text())
    # Include the frozen settings/task tables in the execution identity too.
    base = hashlib.sha256(json.dumps(dict(inputs=manifest['identity'],config=config,plan=payload),sort_keys=True).encode()).hexdigest()
    runtime = execution.make_runtime(config,payload,Path(manifest['data_root']),base,args.smoke)
    if args.action=='task':
        destination = root/'smoke' if args.smoke else root
        print(execution.run_task(runtime,args.index,destination))
    else:
        if args.smoke:
            parser.error('Smoke tasks cannot be collected as full results')
        execution.collect_results(runtime,root)
        print(root/'reduced')


if __name__=='__main__':
    main()
