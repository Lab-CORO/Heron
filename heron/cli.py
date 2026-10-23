"""Command line: robots, stations, datasets, corpora, and `shell-env` for the shell pipeline."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from heron import paths

from heron.errors import HeronConfigError
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
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(str(value))}")


def _prepare(args) -> None:
    from heron.corpus import corpus_dir, load_recipe, prepare
    recipe = load_recipe(args.recipe)
    if not args.submit:
        out = prepare(recipe, force=args.force)
        print(f"corpus written to {out}")
        return
    # Same command as a CPU job: preparing a Corpus is hours of CPU work, too much for
    # a login node and wasted on a GPU allocation.
    if corpus_dir(recipe.name).exists() and not args.force:
        from heron.errors import HeronConfigError
        raise HeronConfigError(f"{corpus_dir(recipe.name)} already exists (use --force to rebuild it from scratch)")
    log_dir = paths.data_root() / "log" / "output"
    command = (f"cd {shlex.quote(str(paths.REPO_ROOT))} && PYTHONPATH={shlex.quote(str(paths.REPO_ROOT))} "
               f"{shlex.quote(sys.executable)} -m heron prepare {shlex.quote(str(recipe.source_file))}"
               + (" --force" if args.force else ""))
    sbatch = ["sbatch", f"--job-name=heron_prepare_{recipe.name}",
              f"--output={log_dir}/heron_prepare_{recipe.name}_%j.txt",
              f"--time={args.time}", f"--mem={args.mem}", f"--cpus-per-task={args.cpus}"]
    if args.account:
        sbatch.append(f"--account={args.account}")
    sbatch.append(f"--wrap={command}")
    print("+ " + " ".join(shlex.quote(a) for a in sbatch))
    if args.dry_run:
        print("(dry-run: not submitted)")
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(sbatch, check=True)


def _train(args) -> None:
    from heron import training
    if args.resume:
        experiment = training.find_experiment(args.resume)
        last = training.prepare_resume(experiment) if not args.dry_run else None
        print(f"resuming {experiment.name} from {last or 'its last checkpoint'}")
        if args.dry_run:
            print("(dry-run: not submitted)")
            return
        print(f"submitted job {training.submit(experiment, args.account, kind='resume')}")
        return
    if not args.recipe:
        from heron.errors import HeronConfigError
        raise HeronConfigError("give a training recipe, or --resume <experiment>")
    recipe = training.load_training_recipe(args.recipe)
    sif = Path(args.sif) if args.sif else paths.data_root() / "capa_unet.sif"
    if not sif.is_file():
        from heron.errors import HeronConfigError
        raise HeronConfigError(f"training image not found: {sif} (use --sif)")
    plan = training.plan(recipe, sif)
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
    print(f"submitted job {training.submit(experiment, args.account)}")


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
    p.add_argument("--submit", action="store_true", help="run as a SLURM CPU job instead of here")
    p.add_argument("--account", help="SLURM account (default: sbatch's)")
    p.add_argument("--time", default="6:00:00")
    p.add_argument("--mem", default="32G")
    p.add_argument("--cpus", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="with --submit: print the sbatch command only")

    p = sub.add_parser("train", help="create an experiment from a training recipe and submit it")
    p.add_argument("recipe", nargs="?")
    p.add_argument("--resume", metavar="EXPERIMENT", help="continue an experiment from its last checkpoint")
    p.add_argument("--account", help="SLURM account (default: sbatch's)")
    p.add_argument("--sif", help="training image (default: <data root>/capa_unet.sif)")
    p.add_argument("--dry-run", action="store_true", help="print config.yaml and job.sh, write nothing")

    p = sub.add_parser("experiments", help="list experiments, or show one")
    p.add_argument("name", nargs="?")

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
        elif args.command == "experiments":
            _experiments(args.name)
    except HeronConfigError as exc:
        print(f"heron: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
