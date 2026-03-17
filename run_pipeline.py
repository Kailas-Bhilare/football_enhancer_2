import subprocess
import argparse
import os


def run_selection(input_video):
    print("\n[STEP 1] Player Selection\n")

    cmd = [
        "python",
        "main.py",
        "--input",
        input_video
    ]

    subprocess.run(cmd)

    if not os.path.exists("selection.json"):
        raise RuntimeError("selection.json not created. Selection step failed.")


def run_render(input_video, output_video):
    print("\n[STEP 2] Rendering Video\n")

    cmd = [
        "python",
        "render_video.py",
        "--input",
        input_video,
        "--output",
        output_video
    ]

    subprocess.run(cmd)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="result.mp4")
    parser.add_argument("--skip-selection", action="store_true")

    args = parser.parse_args()

    if not args.skip_selection:
        run_selection(args.input)
    else:
        print("Skipping selection step")

    run_render(args.input, args.output)

    print("\nPipeline complete →", args.output)


if __name__ == "__main__":
    main()
