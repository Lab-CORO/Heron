from pathlib import Path

import pytest

from heron import check_compatible, list_robots, list_stations, load_robot, load_station
from heron.cli import main
from heron.errors import HeronConfigError
from heron.robot import CUROBO_BUILTIN_DIR


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / f"{name}.yaml"
    path.write_text(text)
    return path


# --- shipped robots and stations ------------------------------------------------

def test_every_shipped_robot_and_station_loads():
    assert set(list_robots()) >= {"m1013", "ur5e", "iiwa", "jaco7"}
    assert set(list_stations()) >= {"leeloo", "bare"}
    for name in list_robots():
        load_robot(name)
    for name in list_stations():
        load_station(name)


def test_leeloo_reproduces_the_historical_crop():
    leeloo = load_station("leeloo")
    assert leeloo.scene_voxels(0.02) == 152
    assert leeloo.crop(0.02, 128) == ((12, 140), (12, 140), (20, 148))


def test_crop_offset_scales_with_voxel_size():
    leeloo = load_station("leeloo")
    assert leeloo.scene_voxels(0.04) == 76
    # 0.16 m is 4 voxels at 0.04
    assert leeloo.crop(0.04, 64) == ((6, 70), (6, 70), (10, 74))


def test_crop_offset_can_be_overridden_per_axis():
    leeloo = load_station("leeloo")
    assert leeloo.crop(0.02, 128, offset=(0.1, -0.1, 0.0)) == ((17, 145), (7, 135), (12, 140))


def test_crop_leaving_the_scene_is_refused():
    with pytest.raises(HeronConfigError, match="leaves the 152-voxel scene"):
        load_station("leeloo").crop(0.02, 128, offset=(0.0, 0.0, 0.5))


def test_builtin_and_default_curobo_configs():
    assert load_robot("ur5e").container_config_file == f"{CUROBO_BUILTIN_DIR}/ur5e.yml"
    assert load_robot("m1013").container_config_file == ""
    assert load_robot("m1013").host_config_dir is None


def test_leeloo_is_not_a_robot_anymore():
    with pytest.raises(HeronConfigError, match="is a station"):
        load_robot("leeloo")


# --- user-supplied files ----------------------------------------------------------

def test_host_robot_config_is_mounted_at_its_own_path(tmp_path):
    (tmp_path / "arm").mkdir()
    (tmp_path / "arm" / "arm.yml").write_text("robot_cfg: {}\n")
    robot = load_robot(str(write(tmp_path, "arm", "name: arm\ncurobo_config: arm/arm.yml\nbase_link: base\nreach: 1.2\n")))
    assert robot.container_config_file == str(tmp_path / "arm" / "arm.yml")
    assert robot.host_config_dir == tmp_path / "arm"


@pytest.mark.parametrize("text, message", [
    ("name: arm\nbase_link: base\n", "missing key"),
    ("name: arm\nbase_link: base\nreach: 1\nbase_lnk: x\n", "unknown key"),
    ("name: other\nbase_link: base\nreach: 1\n", "must match the file name"),
    ("name: arm\nbase_link: base\nreach: 2.5\n", "exceeds 2.0 m"),
    ("name: arm\nbase_link: base\nreach: -1\n", "must be positive"),
    ("name: arm\nbase_link: base\nreach: 1\ncurobo_config: ./missing.yml\n", "file not found"),
])
def test_invalid_robot_files_are_refused(tmp_path, text, message):
    with pytest.raises(HeronConfigError, match=message):
        load_robot(str(write(tmp_path, "arm", text)))


@pytest.mark.parametrize("text, message", [
    ("name: s\n", "missing key"),
    ("name: s\nscene_half_extent: 1\ncrop_offset: [0, 0]\n", "crop_offset must be a list"),
    ("name: s\nscene_half_extent: 1\nworld: nowhere.yaml\n", "world file not found"),
])
def test_invalid_station_files_are_refused(tmp_path, text, message):
    with pytest.raises(HeronConfigError, match=message):
        load_station(str(write(tmp_path, "s", text)))


def test_scene_larger_than_the_master_map_is_refused(tmp_path):
    robot = load_robot(str(write(tmp_path, "small", "name: small\nbase_link: b\nreach: 1.0\n")))
    with pytest.raises(HeronConfigError, match="only covers"):
        check_compatible(robot, load_station("leeloo"))


# --- master maps -------------------------------------------------------------------

def test_new_master_maps_go_under_the_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERON_DATA_ROOT", str(tmp_path))
    robot = load_robot("ur5e")
    assert robot.master_map_dir(0.02) == tmp_path / "master_maps" / "ur5e" / "0.02"
    assert robot.find_master_map_dir(0.02) is None


def test_legacy_master_maps_are_still_found(tmp_path, monkeypatch):
    monkeypatch.setenv("HERON_DATA_ROOT", str(tmp_path))
    (tmp_path / "data_ur5e").mkdir()
    (tmp_path / "data_ur5e" / "master_ik_data0.02.npz").write_bytes(b"")
    assert load_robot("ur5e").find_master_map_dir(0.02) == tmp_path / "data_ur5e"


# --- shell-env, the contract with gen_dataset.sh -------------------------------------

def test_shell_env_for_a_generation_run(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("HERON_DATA_ROOT", str(tmp_path))
    assert main(["shell-env", "--robot", "m1013", "--station", "leeloo", "--voxel", "0.02"]) == 0
    env = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
    assert env["BASE_LINK"] == "base_0"
    assert env["ROBOT_CONFIG_FILE"] == "''"
    assert env["SCENE_HALF_EXTENT"] == "1.52"
    assert env["WORLD_PATH"].endswith("leeloo_world.yaml")
    assert env["MASTER_MAP_DIR"] == str(tmp_path / "master_maps" / "m1013" / "0.02")


def test_shell_env_reports_errors_without_a_traceback(capsys):
    assert main(["shell-env", "--robot", "leeloo", "--voxel", "0.02"]) == 2
    assert "is a station" in capsys.readouterr().err
