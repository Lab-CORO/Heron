"""Command line: `heron robots`, `heron stations`, and `heron shell-env` for the shell pipeline."""

import argparse
import shlex
import sys

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
    except HeronConfigError as exc:
        print(f"heron: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
