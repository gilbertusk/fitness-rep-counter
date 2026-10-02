"""Unit tests for repcount.evaluation.rep_plots."""

import json
import sys

import pytest

from repcount.evaluation import rep_plots


def trace(video_id="squat_1", truth=(1000, 3000), predicted=(1100, 3050), signal=None):
    stamps = list(range(0, 4000, 100))
    return {"video_id": video_id, "label": "squat", "abs_error": abs(len(predicted) - len(truth)),
            "truth_ms": list(truth), "predicted_ms": list(predicted), "timestamps_ms": stamps,
            "signal": signal if signal is not None else [i % 10 / 10 for i in range(len(stamps))]}


def test_the_title_says_whether_the_count_was_right():
    assert "tepat" in rep_plots.figure_title(trace())
    assert "selisih -1" in rep_plots.figure_title(trace(predicted=(1100,)))
    assert "selisih +1" in rep_plots.figure_title(trace(predicted=(1100, 2000, 3050)))


def test_a_figure_is_written_per_trace(tmp_path):
    paths = rep_plots.plot_all({"counter": "generic", "traces": [trace(), trace("squat_2", predicted=())]}, tmp_path)
    assert [p.name for p in paths] == ["generic_squat_1.png", "generic_squat_2.png"]
    assert all(p.stat().st_size > 1000 for p in paths)


def test_a_counter_without_a_signal_still_plots_its_marks(tmp_path):
    """The threshold counter exposes no signal: its trace is all null, and the figure must not break."""
    blank = trace(signal=[None] * 40)
    assert rep_plots.plot_trace(blank, "threshold", tmp_path / "t.png").exists()


def test_the_cli_writes_figures_and_explains_a_missing_file(tmp_path, monkeypatch, capsys):
    traces = tmp_path / "rep_traces_generic_val.json"
    traces.write_text(json.dumps({"counter": "generic", "split": "val", "traces": [trace()]}))
    monkeypatch.setattr(sys, "argv", ["prog", "--traces", str(traces), "--out-dir", str(tmp_path / "fig")])
    rep_plots.main()
    assert "generic_squat_1.png" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["prog", "--traces", str(tmp_path / "nope.json")])
    with pytest.raises(SystemExit, match="run `node tools/eval/evalReps.js` first"):
        rep_plots.main()
