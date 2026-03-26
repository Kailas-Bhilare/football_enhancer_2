import subprocess
import argparse
import os
import shutil
import sys
from pathlib import Path

NORMAL_MODE = "normal"
DEBUG_MODE = "debug"
DEBUG_FRAME_LIMIT = 10
DEBUG_DIR = "debug"


def resolve_run_mode(mode=None, prompt_fn=input):
    if mode in {NORMAL_MODE, DEBUG_MODE}:
        return mode

    prompt = (
        "\nChoose startup mode:\n"
        "1 - run normally\n"
        "2 - run in debug mode\n"
        "> "
    )

    while True:
        try:
            choice = prompt_fn(prompt).strip()
        except EOFError:
            choice = "1"

        if choice in {"", "1"}:
            return NORMAL_MODE

        if choice == "2":
            return DEBUG_MODE

        print("Invalid choice. Enter 1 for normal mode or 2 for debug mode.")


def prepare_debug_directory(debug_dir):
    debug_path = Path(debug_dir)

    if debug_path.exists():
        shutil.rmtree(debug_path)

    debug_path.mkdir(parents=True, exist_ok=True)
    return debug_path


def ensure_selection_exists(selection_path="selection.json"):
    if not Path(selection_path).exists():
        raise FileNotFoundError(
            "selection.json not found. Run normal mode once to create a selection first."
        )


# --------------------------------------------------
# STEP 1: Selection
# --------------------------------------------------
def run_selection(input_video):
    print("\n[STEP 1] Player Selection\n")

    cmd = [
        sys.executable,
        "main.py",
        "--input",
        input_video
    ]

    subprocess.run(cmd, check=True)

    if not os.path.exists("selection.json"):
        raise RuntimeError("selection.json not created. Selection step failed.")


# --------------------------------------------------
# STEP 2: Rendering (UPDATED)
# --------------------------------------------------
def build_render_command(
    input_video,
    output_video,
    debug=False,
    debug_dir=DEBUG_DIR,
    max_frames=DEBUG_FRAME_LIMIT,
):
    cmd = [
        sys.executable,
        "batch_render.py",   # switched from render_video.py
        "--input",
        input_video,
        "--output",
        output_video
    ]

    # Optional debug flags (only if you later support them)
    if debug:
        cmd.extend([
            "--max-frames",
            str(max_frames),
        ])

    return cmd


def run_render(
    input_video,
    output_video,
    debug=False,
    debug_dir=DEBUG_DIR,
    max_frames=DEBUG_FRAME_LIMIT,
):
    print("\n[STEP 2] Rendering Video\n")

    cmd = build_render_command(
        input_video,
        output_video,
        debug=debug,
        debug_dir=debug_dir,
        max_frames=max_frames,
    )

    subprocess.run(cmd, check=True)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="result.mp4")
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument("--mode", choices=[NORMAL_MODE, DEBUG_MODE])

    args = parser.parse_args()
    mode = resolve_run_mode(args.mode)

    # ---------------- DEBUG MODE ----------------
    if mode == DEBUG_MODE:
        ensure_selection_exists()
        debug_path = prepare_debug_directory(DEBUG_DIR)

        print(
            f"Debug mode enabled: using existing selection.json "
            f"and saving {DEBUG_FRAME_LIMIT} frames to {debug_path}/"
        )

        run_render(
            args.input,
            args.output,
            debug=True,
            debug_dir=debug_path,
            max_frames=DEBUG_FRAME_LIMIT,
        )

        print(f"\nDebug run complete → {debug_path}/")
        return

    # ---------------- NORMAL MODE ----------------
    if not args.skip_selection:
        run_selection(args.input)
    else:
        print("Skipping selection step")

    run_render(args.input, args.output)

    print("\nPipeline complete →", args.output)


if __name__ == "__main__":
    main()