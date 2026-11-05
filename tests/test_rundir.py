import os

from train.rundir import latest_checkpoint, make_run_dir


def test_run_dir_and_checkpoint_paths_are_absolute(tmp_path):
    # Regression: orbax refuses relative checkpoint paths (found in the first real-clip run).
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run = make_run_dir({"name": "t", "seed": 0}, runs_root="runs")
        assert run.is_absolute()
        (run / "checkpoints" / "000000000010").mkdir()
        (run / "checkpoints" / "000000000200").mkdir()
        ck = latest_checkpoint(os.path.relpath(run))
        assert ck.is_absolute() and ck.name == "000000000200"
        assert (run / "config.yaml").exists() and (run / "meta.json").exists()
    finally:
        os.chdir(cwd)
