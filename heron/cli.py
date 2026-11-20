"""Command line: robots, stations, datasets, corpora, experiments, TensorRT packages and benchmarks, doctor,
and `shell-env` for the shell pipeline."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from heron import paths

from heron.errors import HeronConfigError
from heron.execution import JobFailed
from heron.robot import list_robots, load_robot
from heron.station import check_compatible, list_stations, load_station


def _show_robot(name: str, voxel_size: float) -> None:
    robot = load_robot(name)
    found = robot.find_master_map_dir(voxel_size)
    print(f"robot          {robot.name}")
    print(f"base_link      {robot.base_link}")
    print(f"reach          ±{robot.reach} m")
    print(f"curobo config  {robot.container_config_file or '<curobo_ros default>'}")
    if robot.host_config_dir:
        print(f"mounted dir    {robot.host_config_dir}")
    print(f"master map     {found if found else 'missing'} (voxel {voxel_size})")


def _show_station(name: str, voxel_size: float, crop_size: int) -> None:
    station = load_station(name)
    n = station.scene_voxels(voxel_size)
    print(f"station        {station.name}")
    print(f"world          {station.world or '<curobo default>'}")
    print(f"cameras        {station.cameras or '-'}")
    print(f"camera frame   {station.camera_frame or '-'}")
    print(f"scene          ±{station.scene_half_extent} m -> {n}³ voxels at {voxel_size}")
    print(f"crop offset    {list(station.crop_offset)} m")
    try:
        crop = station.crop(voxel_size, crop_size)
        flat = " ".join(str(i) for r in crop for i in r)
        print(f"crop {crop_size}³      {flat}")
    except HeronConfigError as exc:
        print(f"crop {crop_size}³      impossible: {exc}")


def _site_account_gpu() -> str:
    from heron.site import load_site
    return load_site().account_gpu or ""


def _shell_env(robot_name: str, station_name: str | None, voxel_size: float) -> None:
    """Print KEY='value' lines for gen_dataset.sh to eval."""
    robot = load_robot(robot_name)
    station = load_station(station_name) if station_name else None
    if station:
        check_compatible(robot, station)
    existing = robot.find_master_map_dir(voxel_size)
    values = {
        "ROBOT_CONFIG_FILE": robot.container_config_file,
        "ROBOT_CONFIG_BIND": robot.host_config_dir or "",
        "BASE_LINK": robot.base_link,
        "MASTER_MAP_DIR": existing or robot.master_map_dir(voxel_size),
        "MASTER_MAP_NEW_DIR": robot.master_map_dir(voxel_size),
        "WORLD_PATH": (station.world if station else None) or "",
        "CAMERAS_TEMPLATE": (station.cameras if station else None) or "",
        "DEFAULT_CAMERA_FRAME": station.camera_frame if station else "",
        "SCENE_HALF_EXTENT": station.scene_half_extent if station else "",
        "ACCOUNT_GPU": _site_account_gpu(),
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(str(value))}")


def _site(args):
    from heron.site import load_site
    return load_site(arch=getattr(args, "arch", None), local=getattr(args, "local", False))


def _prepare(args) -> None:
    from heron.corpus import corpus_dir, load_recipe, prepare
    recipe = load_recipe(args.recipe)
    if not args.submit:
        out = prepare(recipe, force=args.force)
        print(f"corpus written to {out}")
        return
    # Same command as a CPU job: preparing a Corpus is hours of CPU work, too much for
    # a login node and wasted on a GPU allocation.
    from heron.execution import Job, executor, render
    if corpus_dir(recipe.name).exists() and not args.force:
        raise HeronConfigError(f"{corpus_dir(recipe.name)} already exists (use --force to rebuild it from scratch)")
    site = _site(args)
    log_dir = paths.data_root() / "log"
    job = Job(name=f"heron_prepare_{recipe.name}", log_dir=log_dir, log_stem=f"heron_prepare_{recipe.name}",
              command=(f"cd {shlex.quote(str(paths.REPO_ROOT))} && {shlex.quote(sys.executable)} -m heron prepare "
                       f"{shlex.quote(str(recipe.source_file))}" + (" --force" if args.force else "")),
              runtime="host", resources={"time": args.time, "mem": args.mem, "cpus": args.cpus, "gpus": 0},
              env={"PYTHONPATH": paths.REPO_ROOT, "HERON_DATA_ROOT": paths.data_root()})
    script_text = render(job, site)
    runner = executor(site, args.account)
    script = log_dir / f"heron_prepare_{recipe.name}.sh"
    if args.dry_run:
        print(script_text)
        if runner.name == "slurm":
            print("+ " + " ".join(runner.command(script, gpu=False)))
        print("(dry-run: nothing written, not submitted)")
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    script.write_text(script_text)
    print(f"{runner.name} job {runner.submit(script, gpu=False, log_dir=log_dir, log_stem=job.log_stem)}")


def _train(args) -> None:
    from heron import training
    from heron.execution import executor
    site = _site(args)
    runner = executor(site, args.account)
    if args.resume:
        experiment = training.find_experiment(args.resume)
        last = training.prepare_resume(experiment) if not args.dry_run else None
        print(f"resuming {experiment.name} from {last or 'its last checkpoint'}")
        if args.dry_run:
            print("(dry-run: not submitted)")
            return
        print(f"{runner.name} job {training.submit(experiment, runner, kind='resume')}")
        return
    if not args.recipe:
        raise HeronConfigError("give a training recipe, or --resume <experiment>")
    recipe = training.load_training_recipe(args.recipe)
    plan = training.plan(recipe, site)
    if args.dry_run:
        import yaml
        print(f"# experiment: {plan.experiment}\n# config.yaml")
        print(yaml.safe_dump(plan.config, sort_keys=False))
        print("# job.sh")
        print(plan.job)
        print("(dry-run: nothing written, not submitted)")
        return
    experiment = training.create(plan)
    print(f"experiment {experiment}")
    if not args.no_evaluate:
        # Written before training starts, so a local run evaluates right after it.
        from heron.evaluation.jobs import EVAL_SCRIPT, write_eval_job
        write_eval_job(experiment, site)
    job = training.submit(experiment, runner)
    print(f"{runner.name} training job {job}")
    if not args.no_evaluate:
        eval_job = training.submit(experiment, runner, kind="evaluate", script=EVAL_SCRIPT, after=job)
        print(f"{runner.name} evaluation job {eval_job}" + (f" (runs if training {job} succeeds)"
                                                           if runner.name == "slurm" else ""))


def _evaluate(args) -> None:
    from heron import training
    from heron.evaluation import jobs
    from heron.execution import executor
    experiment = training.find_experiment(args.experiment)
    if args.figures:
        jobs.draw_figures(experiment)
        return
    site = _site(args)
    resources = {"time": args.time} if args.time else None
    if args.dry_run:
        print(jobs.eval_job_script(experiment, args.checkpoint, jobs.select_parts(experiment, args.part),
                                   site, resources, args.precision, args.samples))
        print("(dry-run: nothing written, not submitted)")
        return
    jobs.write_eval_job(experiment, site, args.checkpoint, args.part, resources, args.precision, args.samples)
    runner = executor(site, args.account)
    job = training.submit(experiment, runner, kind="evaluate", script=jobs.EVAL_SCRIPT)
    print(f"{runner.name} evaluation job {job}; results in {jobs.results_path(experiment)}")
    print(f"then draw the figures with: heron evaluate {experiment.name} --figures")


def _export(args) -> None:
    from heron import training
    from heron.inference.package import export
    experiment = training.find_experiment(args.experiment)
    out = export(experiment, Path(args.out) if args.out else None, args.checkpoint,
                 Path(args.calibration) if args.calibration else None, args.calibration_volumes, args.force,
                 args.voxel_size, tuple(args.origin) if args.origin else None, args.reference_scenes)
    print(f"package {out}")
    print(f"time it on this machine's GPU with: heron benchmark {out} --precision fp32 fp16 int8")


def _benchmark(args) -> None:
    from heron.execution import executor
    from heron.inference import jobs
    from heron.inference.package import find_package, load
    package = find_package(args.package)
    site = _site(args)
    options = dict(samples=args.samples, warmup=args.warmup, rebuild=args.rebuild,
                   resources={"time": args.time} if args.time else None)
    if args.dry_run:
        from heron.execution import render
        print(render(jobs.benchmark_job(package, args.precision, site, **options), site))
        print("(dry-run: nothing written, not submitted)")
        return
    script = jobs.write_benchmark_job(package, args.precision, site, **options)
    runner = executor(site, args.account)
    job = runner.submit(script, gpu=True, log_dir=package / "logs", log_stem="benchmark")
    print(f"{runner.name} benchmark job {job}; results in {jobs.results_dir(load(package)['experiment'])}/")


def _doctor(args) -> int:
    from heron import doctor
    from heron.execution import executor
    if args.report:
        return 0 if doctor.report(Path(args.report), keep=args.keep) else 1
    site = _site(args)
    runner = executor(site, args.account)
    job = doctor.start(site, runner, keep=args.keep)
    if runner.name == "slurm":
        print(f"slurm job {job}: the report will be at the end of "
              f"{paths.data_root() / 'log'}/heron_doctor_{job}.out")
    return 0


def _experiments(name: str | None) -> None:
    from heron import training
    if name:
        s = training.summary(training.find_experiment(name))
        for key, value in s.items():
            print(f"{key:12} {value}")
        return
    for experiment in training.list_experiments():
        s = training.summary(experiment)
        score = f"{s['best_score']:.4f}" if s["best_score"] is not None else "-"
        ckpt = s["checkpoints"].get("last", "no checkpoint")
        print(f"{s['name']:40} corpus={s['corpus']:20} best={score:8} last={ckpt}")


def _execution_options(p) -> None:
    p.add_argument("--local", action="store_true", help="run here, in the foreground, instead of with SLURM")
    p.add_argument("--arch", choices=["amd64", "aarch64"], help="force the image architecture (default: this machine's)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="heron")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("robots", help="list robots, or show one")
    p.add_argument("name", nargs="?")
    p.add_argument("--voxel", type=float, default=0.02)

    p = sub.add_parser("stations", help="list stations, or show one with its derived grid and crop")
    p.add_argument("name", nargs="?")
    p.add_argument("--voxel", type=float, default=0.02)
    p.add_argument("--crop-size", type=int, default=128)

    p = sub.add_parser("shell-env", help="robot and station facts as shell assignments (used by gen_dataset.sh)")
    p.add_argument("--robot", required=True)
    p.add_argument("--station")
    p.add_argument("--voxel", type=float, required=True)

    p = sub.add_parser("dataset", help="datasets (raw generator output)")
    dsub = p.add_subparsers(dest="dataset_command", required=True)
    d = dsub.add_parser("describe", help="write the dataset.yaml manifest of a dataset directory")
    d.add_argument("path")
    d.add_argument("--robot", required=True)
    d.add_argument("--station", required=True)
    d.add_argument("--mode", required=True, choices=["real", "synthetic", "scene"])
    d.add_argument("--voxel", type=float, required=True)
    d.add_argument("--source", default="", help="rosbag or scene file name (real: defaults to the directory name)")
    d.add_argument("--force", action="store_true", help="overwrite an existing manifest")

    p = sub.add_parser("prepare", help="build a corpus from a recipe file")
    p.add_argument("recipe")
    p.add_argument("--force", action="store_true", help="delete and rebuild an existing corpus")
    p.add_argument("--submit", action="store_true", help="run as a job (SLURM CPU job, or here with --local)")
    p.add_argument("--account", help="SLURM account (default: slurm.account_cpu in site.yaml, else sbatch's)")
    p.add_argument("--time", default="6:00:00")
    p.add_argument("--mem", default="32G")
    p.add_argument("--cpus", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="with --submit: print the job, write nothing")
    _execution_options(p)

    p = sub.add_parser("train", help="create an experiment from a training recipe and submit it")
    p.add_argument("recipe", nargs="?")
    p.add_argument("--resume", metavar="EXPERIMENT", help="continue an experiment from its last checkpoint")
    p.add_argument("--account", help="SLURM account (default: slurm.account_gpu in site.yaml, else sbatch's)")
    p.add_argument("--dry-run", action="store_true", help="print config.yaml and job.sh, write nothing")
    p.add_argument("--no-evaluate", action="store_true", help="do not chain the evaluation job after training")
    _execution_options(p)

    p = sub.add_parser("evaluate", help="measure an experiment on its corpus (val and test parts)")
    p.add_argument("experiment")
    p.add_argument("--part", action="append", help="only this part (val, test:<name>); repeatable")
    p.add_argument("--checkpoint", default="best", choices=["best", "last"])
    p.add_argument("--account", help="SLURM account (default: slurm.account_gpu in site.yaml, else sbatch's)")
    p.add_argument("--time", help="job time limit (default 2:00:00)")
    p.add_argument("--dry-run", action="store_true", help="print the job, write nothing")
    _execution_options(p)
    p.add_argument("--figures", action="store_true", help="draw the figures of a finished measurement, here")
    p.add_argument("--precision", nargs="+", default=[], choices=["fp32", "fp16", "int8"],
                   help="also time and measure these TensorRT engines against PyTorch (exports the package if needed)")
    p.add_argument("--samples", type=int, help="with --precision: volumes timed per run (default 1000)")

    p = sub.add_parser("export", help="package an experiment for inference elsewhere: checkpoint, ONNX, int8 sample")
    p.add_argument("experiment")
    p.add_argument("--checkpoint", default="best", choices=["best", "last"])
    p.add_argument("--out", help="package directory (default: <experiment>/package)")
    p.add_argument("--calibration", metavar="DIR",
                   help="train volumes for int8 calibration (default: the corpus's train part)")
    p.add_argument("--calibration-volumes", type=int, default=200, help="how many of them to package (default 200)")
    p.add_argument("--force", action="store_true", help="replace an existing package, engines included")
    p.add_argument("--voxel-size", type=float, metavar="M",
                   help="voxel size of the network's input, when the volumes' attributes do not say it")
    p.add_argument("--origin", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="position of voxel [0, 0, 0] from the robot base (m), with --voxel-size")
    p.add_argument("--reference-scenes", type=int, default=5,
                   help="calibration volumes also packaged as reference Scenes with their PyTorch maps, "
                        "against which a robot checks its own engine (default 5)")

    p = sub.add_parser("benchmark", help="time a package on a GPU: PyTorch fp32, then TensorRT engines")
    p.add_argument("package", help="package directory, or an experiment (its package)")
    p.add_argument("--precision", nargs="+", default=["fp32", "fp16", "int8"], choices=["fp32", "fp16", "int8"],
                   help="TensorRT precisions to build and time (default: all three)")
    p.add_argument("--samples", type=int, default=1000, help="volumes timed per run (default 1000)")
    p.add_argument("--warmup", type=int, default=50, help="volumes run before timing (default 50)")
    p.add_argument("--rebuild", action="store_true", help="build the engines again even if cached")
    p.add_argument("--account", help="SLURM account (default: slurm.account_gpu in site.yaml, else sbatch's)")
    p.add_argument("--time", help="job time limit (default 2:00:00)")
    p.add_argument("--dry-run", action="store_true", help="print the job, write nothing")
    _execution_options(p)

    p = sub.add_parser("experiments", help="list experiments, or show one")
    p.add_argument("name", nargs="?")

    p = sub.add_parser("doctor", help="run a tiny pipeline end to end on this site and report what works")
    p.add_argument("--account", help="SLURM account (default: slurm.account_gpu in site.yaml, else sbatch's)")
    p.add_argument("--keep", action="store_true", help="keep the run's files instead of deleting them")
    p.add_argument("--report", metavar="DIR", help=argparse.SUPPRESS)
    _execution_options(p)

    args = parser.parse_args(argv)
    try:
        if args.command == "robots":
            if args.name:
                _show_robot(args.name, args.voxel)
            else:
                print("\n".join(list_robots()))
        elif args.command == "stations":
            if args.name:
                _show_station(args.name, args.voxel, args.crop_size)
            else:
                print("\n".join(list_stations()))
        elif args.command == "shell-env":
            _shell_env(args.robot, args.station, args.voxel)
        elif args.command == "dataset":
            from heron.dataset import describe
            manifest = describe(Path(args.path), args.robot, args.station, args.mode, args.voxel,
                                source=args.source, force=args.force)
            print(f"wrote {manifest}")
        elif args.command == "prepare":
            _prepare(args)
        elif args.command == "train":
            _train(args)
        elif args.command == "evaluate":
            _evaluate(args)
        elif args.command == "export":
            _export(args)
        elif args.command == "benchmark":
            _benchmark(args)
        elif args.command == "experiments":
            _experiments(args.name)
        elif args.command == "doctor":
            return _doctor(args)
    except HeronConfigError as exc:
        print(f"heron: {exc}", file=sys.stderr)
        return 2
    except JobFailed as exc:
        print(f"heron: job failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
