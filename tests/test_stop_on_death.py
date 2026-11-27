"""A generation launch ends, and fails, when a process it needs dies (launch/stop_on_death.py).

The behaviour itself needs ROS 2 launch, so it is checked in the image; here, that every
Jazzy launch file hands each process it needs to stop_on_death.
"""
import re
from pathlib import Path

import pytest

LAUNCH = Path(__file__).resolve().parents[1] / "data_generation" / "launch"

NEEDED = {
    "generate_data.launch.py": ["curobo_trajectory_planner", "obstacle_adder", "generate_data"],
    "generate_real_data.launch.py": ["curobo_trajectory_planner", "obstacle_adder", "generate_data",
                                     "ros2 bag play"],
    "create_reachability_map.launch.py": ["curobo_trajectory_planner"],
}


@pytest.mark.parametrize("launch_file", sorted(NEEDED))
def test_every_needed_process_stops_the_launch_when_it_dies(launch_file):
    text = (LAUNCH / launch_file).read_text()
    assert "from stop_on_death import stop_on_death" in text
    wired = re.findall(r"on_exit=stop_on_death\('([^']+)'\)", text)
    assert sorted(wired) == sorted(NEEDED[launch_file])
