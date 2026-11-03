from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import gmm_migt
from gmm_migt import RandomSplitter
from gmm_migt.cli import main


def test_cli_run(dataset, tmp_path, capsys):
    out = str(tmp_path / "out")
    rep = str(tmp_path / "rep.json")
    code = main([dataset, out, "--thesis", "--quiet", "--n-components", "3", "--pca-variance", "0.9",
                 "--report", rep])
    assert code == 0
    assert os.path.isdir(os.path.join(out, "val", "alpha"))
    cfg = json.load(open(rep, encoding="utf-8"))["config"]
    assert (cfg["n_components"], cfg["pca_variance"], cfg["min_bin"], cfg["val"]) == (3, 0.9, 10, 0.1)
    assert "TOTAL" in capsys.readouterr().out


def test_cli_dry_run_and_options(dataset, tmp_path):
    assert main([dataset, "--dry-run", "--quiet", "--no-gmm", "--no-pca", "--val", "none",
                 "--resize-to", "24x24", "--min-bin", "5"]) == 0
    assert not os.path.exists(str(tmp_path / "out"))


def test_cli_errors(dataset, capsys):
    assert main([dataset, "--quiet"]) == 2
    assert main([dataset, "--dry-run", "--train", "0.5", "--quiet"]) == 2
    assert "sum to 1" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([dataset, "--dry-run", "--resize-to", "abc"])


def test_module_entry_point_version():
    r = subprocess.run([sys.executable, "-m", "gmm_migt", "--version"], capture_output=True, text=True)
    assert r.returncode == 0 and gmm_migt.__version__ in r.stdout


def test_random_splitter(dataset, tmp_path):
    rs = RandomSplitter(dataset, train=0.5, test=0.4, val=0.1, seed=3, verbose=False)
    res = rs.run(str(tmp_path / "out"))
    assert res.method == "Random"
    c = res.classes["alpha"].counts
    assert (c["train"], c["test"], c["val"]) == (30, 24, 6)
    again = RandomSplitter(dataset, train=0.5, test=0.4, val=0.1, seed=3, verbose=False).plan()
    assert res.splits() == again.splits()
    two = RandomSplitter(dataset, verbose=False).plan()
    assert set(two.split_names) == {"train", "test"} and two.classes["beta"].counts["test"] == 6
    with pytest.raises(ValueError):
        RandomSplitter(dataset, train=0.9, test=0.2)


def test_plot_histogram(dataset):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from gmm_migt.plotting import plot_class_histogram

    res = gmm_migt.GMMMIGTSplitter.from_thesis(dataset, verbose=False).plan()
    ax = plot_class_histogram(res.classes["alpha"])
    assert len(ax.patches) == len(res.classes["alpha"].bins)
