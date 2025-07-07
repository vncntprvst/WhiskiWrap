import runpy
import sys
import types
from pathlib import Path
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
TRACE_SCRIPT = REPO_DIR / "trace_whiskers.py"
PIPELINE_SCRIPT = REPO_DIR / "whisker_tracking_pipeline.py"


STUB_MODULES = {
    "WhiskiWrap": types.SimpleNamespace(
        FFmpegReader=lambda *a, **k: types.SimpleNamespace(
            frame_width=10, frame_height=10, frame_rate=30
        ),
        interleaved_reading_and_tracing=lambda *a, **k: None,
    ),
    "numpy": types.ModuleType("numpy"),
    "wwutils": types.ModuleType("wwutils"),
}
for sub in [
    "load_whisker_data",
    "whiskerpad",
    "combine_sides",
    "reclassify",
    "unet_classifier",
    "plot_overlay",
]:
    setattr(STUB_MODULES["wwutils"], sub, types.ModuleType(sub))


def run_script_with_stubs(script_path, argv, capsys, monkeypatch):
    """Execute a script with stubbed modules to avoid heavy deps."""
    for name, mod in STUB_MODULES.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(sys, "argv", [str(script_path)] + argv)
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(script_path), run_name="__main__")
    captured = capsys.readouterr()
    return excinfo.value.code, captured.out + captured.err


def test_trace_whiskers_help(capsys, monkeypatch):
    code, output = run_script_with_stubs(TRACE_SCRIPT, ['--help'], capsys, monkeypatch)
    assert code == 0
    assert 'usage' in output.lower()


def test_trace_whiskers_missing_file(capsys, monkeypatch):
    code, output = run_script_with_stubs(TRACE_SCRIPT, ['-v', 'nonexistent.mp4'], capsys, monkeypatch)
    assert code == 1
    assert 'not found' in output.lower() or 'error' in output.lower()


def test_pipeline_version(capsys, monkeypatch):
    code, output = run_script_with_stubs(PIPELINE_SCRIPT, ['--version'], capsys, monkeypatch)
    assert code == 0
    assert 'pipeline' in output.lower()


def test_pipeline_missing_input(capsys, monkeypatch):
    code, output = run_script_with_stubs(PIPELINE_SCRIPT, ['nonexistent.mp4'], capsys, monkeypatch)
    assert code == 1
    assert 'not found' in output.lower() or 'error' in output.lower()
