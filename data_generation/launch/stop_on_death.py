"""End a generation launch when a process it needs dies, and say why.

Without this, the launch keeps running when, say, the cuRobo node dies at start-up:
scene_manager and generate_data wait for its services until the SLURM time limit, and
the task ends in a timeout with no word of the real error.

The reason goes to share/data_generation/data/generation_failed, the file scene_manager
writes when it gives up; run_generation.sh prints it and exits 1.
"""
import os

from launch.actions import Shutdown


def failed_mark():
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory("data_generation"), "data", "generation_failed")


def stop_on_death(name, mark=None):
    """on_exit for a process the task needs: its exit before the end fails the task."""
    def on_exit(event, context):
        if context.is_shutdown:  # the launch is already ending (last Scene done, time limit)
            return None
        why = f"{name} exited (code {event.returncode}) before the task was done"
        try:
            with open(mark or failed_mark(), "w") as f:
                f.write(why + "\n")
        except OSError:
            pass  # the shutdown below still ends the task
        return [Shutdown(reason=why)]
    return on_exit
