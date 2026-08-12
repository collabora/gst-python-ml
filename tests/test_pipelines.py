import subprocess
import os
import signal
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

# Seconds to let a pipeline run. Overridable because a local LLM generating a
# few hundred tokens takes minutes, not seconds.
PIPELINE_TIMEOUT = int(os.environ.get("PIPELINE_TIMEOUT", "30"))

# Only these mean the pipeline broke. Warnings are not fatal: plugins unrelated
# to the pipeline warn during setup and would fail a run that went fine.
FATAL_LOG_PATTERNS = (
    re.compile(r"^ERROR:.*", re.MULTILINE),
    re.compile(r"^WARNING: erroneous pipeline.*", re.MULTILINE),
    re.compile(r"^\S+ +\S+ +\S+ +ERROR +python .*", re.MULTILINE),
    # g2g's own failures: the launcher's parse / run errors and the rewrite's.
    re.compile(r"^(?:parse|pipeline) error:.*", re.MULTILINE),
    re.compile(r"^pyml-launch: (?:no|unknown) .*", re.MULTILINE),
)

BACKEND = os.environ.get("PYML_BACKEND", "gst").lower()

# The launcher the README examples name. Runs from a tmp dir, so it is spelled
# absolute here.
LAUNCHER = f"python {BASE_DIR / 'pyml-launch.py'}"

if BACKEND == "gst" and not shutil.which("gst-launch-1.0"):
    raise RuntimeError("gst-launch-1.0 not found in PATH. Please install GStreamer.")


# Read pipelines from README and modify for frame limit
def get_pipelines_from_readme():
    readme_path = BASE_DIR / "README.md"
    if not readme_path.exists():
        pytest.fail("README.md not found in project root")

    with open(readme_path, "r") as f:
        content = f.read()

    # Match pyml-launch commands, accounting for Markdown backticks
    pipeline_pattern = (
        r"(?:`)?\s*(python pyml-launch\.py\s+.*?)(?:`)?(?=\n\n|\n\s*\n|$)"
    )
    pipelines = re.findall(pipeline_pattern, content, re.DOTALL)

    modified_pipelines = []
    for pipeline in pipelines:
        pipeline = pipeline.strip().strip("`")
        print(f"Raw pipeline after stripping: {pipeline}")

        if not pipeline.startswith("python pyml-launch.py"):
            print(f"Skipping invalid pipeline: {pipeline}")
            continue
        pipeline = pipeline.replace("python pyml-launch.py", LAUNCHER, 1)

        parts = pipeline.split("!")

        # A `filesrc` run has no equivalent cap: these mp4s carry `moov` at the
        # end, so bounding the source by bytes leaves the decoder with no index.
        # Those pipelines run until PIPELINE_TIMEOUT instead.
        for i, part in enumerate(parts):
            part_clean = part.strip()
            if "videotestsrc" in part_clean:
                if "num-buffers=" not in part_clean:
                    # Append num-buffers=100 as part of the element, not after !
                    parts[i] = f"{part_clean} num-buffers=100"
                else:
                    parts[i] = re.sub(r"num-buffers=\d+", "num-buffers=100", part_clean)
                break

        modified_pipeline = " ! ".join(parts).strip()
        print(f"Modified pipeline: {modified_pipeline}")
        modified_pipelines.append(modified_pipeline)
    return modified_pipelines


PIPELINES = get_pipelines_from_readme()


def end_process_group(process):
    """Stop the pipeline and everything it started.

    `shell=True` makes the shell the direct child, so signalling the process
    alone leaves the launcher and its window behind.
    """
    try:
        group = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    os.killpg(group, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(group, signal.SIGKILL)


def absolutize_project_inputs(pipeline):
    """Point a pipeline's relative input paths at the project directory.

    Lets the pipeline run from the test's tmp dir so its output lands there.
    Only values that already name a file are rewritten, so output paths and
    caps strings are left alone.
    """

    def rewrite(match):
        key, value = match.group(1), match.group(2)
        candidate = BASE_DIR / value.strip('"')
        return f"{key}={candidate}" if candidate.is_file() else match.group(0)

    return re.sub(r"([\w-]+)=([^\s!]+)", rewrite, pipeline)


@pytest.mark.serial
@pytest.mark.parametrize("pipeline", PIPELINES, ids=lambda p: p)
def test_pipeline(pipeline, tmp_path):
    """
    Run a README pipeline and check its log for errors.

    A pipeline still running at `PIPELINE_TIMEOUT` passes: only `videotestsrc`
    takes a frame cap, so a file-backed one runs as long as its media lasts.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.sync()
    pipeline = absolutize_project_inputs(pipeline)
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
    # Colour escapes land in the log file and break matching on the level field.
    env["GST_DEBUG_NO_COLOR"] = "1"

    # Run the pipeline
    ran_to_the_cap = False
    try:
        with open(log_file, "w") as log:
            # Own process group: the shell is not the pipeline, it is the
            # launcher's parent, so killing the group is what stops the run.
            # Terminating the shell alone leaves the launcher holding a window
            # and the GPU until the machine is rebooted.
            process = subprocess.Popen(
                pipeline,
                shell=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=tmp_path,
                env=env,
                start_new_session=True,
            )
            process.wait(timeout=PIPELINE_TIMEOUT)
            return_code = process.returncode
    except subprocess.TimeoutExpired:
        # Still running at the cap, which is what a healthy uncapped pipeline
        # does: the media outlasts any timeout worth waiting. The log below says
        # whether it was working, so the run is judged on that, not on exiting.
        end_process_group(process)
        ran_to_the_cap = True
        return_code = None
    except Exception as e:
        end_process_group(process)
        pytest.fail(
            f"Failed to execute pipeline: {e}. Full pipeline: {pipeline}. See {log_file}"
        )

    # Check logs for errors
    if not log_file.exists():
        pytest.fail(f"Log file {log_file} was not created. Full pipeline: {pipeline}")
    with open(log_file, "r") as log:
        log_content = log.read()
    failures = [m.group(0) for p in FATAL_LOG_PATTERNS for m in p.finditer(log_content)]
    if failures:
        reported = "\n".join(failures)
        pytest.fail(
            f"Errors found in pipeline:\n{reported}\nFull pipeline: {pipeline}\nSee {log_file}"
        )

    # Without this a pipeline that never left PAUSED passes on an empty log.
    if "Setting pipeline to PLAYING" not in log_content:
        pytest.fail(
            f"Pipeline never reached PLAYING. Full pipeline: {pipeline}. See {log_file}"
        )

    # Check exit code
    if not ran_to_the_cap and return_code != 0:
        if (
            "End-Of-Stream" not in log_content
            and "reached end of stream" not in log_content
        ):
            pytest.fail(
                f"Pipeline failed with exit code {return_code}. Full pipeline: {pipeline}. See {log_file}"
            )

    ending = f"ran the full {PIPELINE_TIMEOUT}s" if ran_to_the_cap else "ran to the end"
    print(f"Pipeline {ending} with no errors: {pipeline}")


def test_pipelines_found():
    """Ensure at least one pipeline was found in README."""
    if not PIPELINES:
        pytest.fail("No pyml-launch pipelines found in README.md")
    print(f"Found {len(PIPELINES)} pipelines to test")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
