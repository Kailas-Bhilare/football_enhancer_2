from pathlib import Path

from run_pipeline import (
    DEBUG_MODE,
    NORMAL_MODE,
    build_render_command,
    prepare_debug_directory,
    resolve_run_mode,
)


def test_resolve_run_mode_accepts_debug_choice():
    mode = resolve_run_mode(prompt_fn=lambda _: "2")

    assert mode == DEBUG_MODE


def test_resolve_run_mode_defaults_to_normal_on_empty_input():
    mode = resolve_run_mode(prompt_fn=lambda _: "")

    assert mode == NORMAL_MODE


def test_prepare_debug_directory_recreates_empty_folder(tmp_path):
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    (debug_dir / "old_frame.png").write_text("stale")

    prepared = prepare_debug_directory(debug_dir)

    assert prepared == debug_dir
    assert prepared.exists()
    assert list(prepared.iterdir()) == []


def test_build_render_command_enables_debug_arguments():
    cmd = build_render_command(
        "input.mp4",
        "output.mp4",
        debug=True,
        debug_dir=Path("debug"),
        max_frames=10,
    )

    assert cmd == [
        "python",
        "render_video.py",
        "--input",
        "input.mp4",
        "--output",
        "output.mp4",
        "--debug",
        "--debug-dir",
        "debug",
        "--max-frames",
        "10",
    ]
