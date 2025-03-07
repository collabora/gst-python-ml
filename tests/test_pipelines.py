import subprocess
import os
import re
import pytest
from pathlib import Path
import shutil
import uuid
import stat
import time

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "tests" / "logs"

# Check if gst-launch-1.0 is available
if not shutil.which("gst-launch-1.0"):
    raise RuntimeError("gst-launch-1.0 not found in PATH. Please install GStreamer.")


# Read pipelines from README and modify for frame limit
def get_pipelines_from_readme():
    readme_path = BASE_DIR / "README.md"
    if not readme_path.exists():
        pytest.fail("README.md not found in project root")

    with open(readme_path, "r") as f:
        content = f.read()

    # Match gst-launch-1.0 commands, accounting for Markdown backticks
    pipeline_pattern = (
        r"(?:`)?\s*(GST_DEBUG=\d+\s+gst-launch-1\.0\s+.*?)(?:`)?(?=\n\n|\n\s*\n|$)"
    )
    pipelines = re.findall(pipeline_pattern, content, re.DOTALL)

    modified_pipelines = []
    for pipeline in pipelines:
        pipeline = pipeline.strip().strip("`")
        print(f"Raw pipeline after stripping: {pipeline}")

        if not pipeline.startswith(("GST_DEBUG=", "gst-launch-1.0")):
            print(f"Skipping invalid pipeline: {pipeline}")
            continue

        parts = pipeline.split("!")
        modified = False

        for i, part in enumerate(parts):
            part_clean = part.strip()
            if "videotestsrc" in part_clean:
                if "num-buffers=" not in part_clean:
                    # Append num-buffers=100 as part of the element, not after !
                    parts[i] = f"{part_clean} num-buffers=100"
                else:
                    parts[i] = re.sub(r"num-buffers=\d+", "num-buffers=100", part_clean)
                modified = True
                break

        if not modified:
            for i, part in enumerate(parts):
                part_clean = part.strip()
                if "filesrc" in part_clean:
                    parts.insert(i + 1, "queue max-size-buffers=100 leaky=upstream")
                    modified = True
                    break

        if not modified:
            print(f"Warning: No filesrc or videotestsrc found in pipeline: {pipeline}")

        modified_pipeline = " ! ".join(parts).strip()
        print(f"Modified pipeline: {modified_pipeline}")
        modified_pipelines.append(modified_pipeline)
    return modified_pipelines


PIPELINES = get_pipelines_from_readme()


@pytest.mark.serial
@pytest.mark.parametrize("pipeline", PIPELINES, ids=lambda p: p)
def test_pipeline(pipeline, tmp_path):
    """
    Test a GStreamer pipeline for 100 frames, checking for errors, with latency tracing.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.sync()
    unique_id = uuid.uuid4().hex[:8]
    log_file = LOG_DIR / f"test_{unique_id}.log"

    print(f"Testing pipeline: {pipeline}")
    print(f"Log file: {log_file}")

    # Check if input file exists for filesrc
    match = re.search(r"filesrc location=([^\s!]+)", pipeline)
    if match:
        file_path = Path(match.group(1))
        if not file_path.is_absolute():
            file_path = BASE_DIR / file_path
        if not file_path.exists():
            pytest.fail(f"Input file not found: {file_path}. Full pipeline: {pipeline}")

    # Verify log directory state
    if not LOG_DIR.exists():
        pytest.fail(
            f"Log directory {LOG_DIR} does not exist after mkdir. Check permissions."
        )
    if not os.access(str(LOG_DIR), os.W_OK):
        perms = oct(stat.S_IMODE(os.stat(LOG_DIR).st_mode))
        pytest.fail(f"No write permission for {LOG_DIR}. Current perms: {perms}")

    print(f"Log dir exists: {LOG_DIR.exists()}")
    print(f"Log dir writable: {os.access(str(LOG_DIR), os.W_OK)}")
    print(f"Log dir contents: {list(LOG_DIR.iterdir())}")

    # Create the log file
    try:
        fd = os.open(str(log_file), os.O_CREAT | os.O_WRONLY, 0o666)
        os.close(fd)
        print(f"Log file {log_file} created successfully with os.open")
    except Exception as e:
        pytest.fail(f"Failed to create log file {log_file} with os.open: {e}")

    time.sleep(0.1)

    # Set up environment with latency tracer
    env = os.environ.copy()
    env["GST_TRACERS"] = "latency"

    # Run the pipeline
    try:
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                pipeline,
                shell=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=BASE_DIR,
                env=env,
            )
            process.wait(timeout=30)
            return_code = process.returncode
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        pytest.fail(
            f"Pipeline timed out after 30s. Full pipeline: {pipeline}. See {log_file}"
        )
    except Exception as e:
        pytest.fail(
            f"Failed to execute pipeline: {e}. Full pipeline: {pipeline}. See {log_file}"
        )

    # Check logs for errors
    if not log_file.exists():
        pytest.fail(f"Log file {log_file} was not created. Full pipeline: {pipeline}")
    with open(log_file, "r") as log:
        log_content = log.read()
        error_lines = [
            line
            for line in log_content.splitlines()
            if "ERROR" in line or "WARN" in line
        ]
        if error_lines:
            pytest.fail(
                f"Errors/Warnings found in pipeline:\n{''.join(error_lines)}\nFull pipeline: {pipeline}\nSee {log_file}"
            )

    # Check exit code
    if return_code != 0:
        if (
            "End-Of-Stream" not in log_content
            and "reached end of stream" not in log_content
        ):
            pytest.fail(
                f"Pipeline failed with exit code {return_code}. Full pipeline: {pipeline}. See {log_file}"
            )

    print(f"Pipeline processed 100 frames successfully: {pipeline}")


def test_pipelines_found():
    """Ensure at least one pipeline was found in README."""
    if not PIPELINES:
        pytest.fail("No gst-launch-1.0 pipelines found in README.md")
    print(f"Found {len(PIPELINES)} pipelines to test")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
