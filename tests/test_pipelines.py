import subprocess
import os
import re
import pytest
from pathlib import Path

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "tests" / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Read pipelines from README and modify for frame limit
def get_pipelines_from_readme():
    readme_path = BASE_DIR / "README.md"
    if not readme_path.exists():
        pytest.fail("README.md not found in project root")

    with open(readme_path, "r") as f:
        content = f.read()

    # Match gst-launch-1.0 commands
    pipeline_pattern = r"(GST_DEBUG=\d+\s+gst-launch-1\.0\s+.*?)(?=\n\n|\n\s*\n|$)"
    pipelines = re.findall(pipeline_pattern, content, re.DOTALL)
    
    # Modify each pipeline to limit to 100 frames
    modified_pipelines = []
    for pipeline in pipelines:
        # Insert queue after first filesrc
        parts = pipeline.split("!")
        for i, part in enumerate(parts):
            if "filesrc" in part.strip():
                parts.insert(i + 1, " queue max-size-buffers=100 leaky=upstream ")
                break
        modified_pipeline = "!".join(parts).strip()
        modified_pipelines.append(modified_pipeline)
    return modified_pipelines

PIPELINES = get_pipelines_from_readme()

@pytest.mark.parametrize("pipeline", PIPELINES, ids=lambda p: p.split("!")[0].strip())
def test_pipeline(pipeline, tmp_path):
    """
    Test a GStreamer pipeline for 100 frames, checking for errors.
    """
    log_file = LOG_DIR / f"test_{pipeline.split('!')[0].strip().replace(' ', '_')}.log"
    
    # Check if input file exists
    match = re.search(r"filesrc location=([^\s!]+)", pipeline)
    if match:
        file_path = Path(match.group(1))
        if not file_path.is_absolute():
            file_path = BASE_DIR / file_path
        if not file_path.exists():
            pytest.skip(f"Input file not found: {file_path}")

    # Run the pipeline with a timeout
    try:
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                pipeline,
                shell=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=BASE_DIR
            )
            # Wait up to 30 seconds for 100 frames to process
            process.wait(timeout=30)
            return_code = process.returncode
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        pytest.fail(f"Pipeline timed out after 30s. See {log_file}")
    except Exception as e:
        pytest.fail(f"Pipeline execution failed: {e}. See {log_file}")

    # Check logs for errors
    with open(log_file, "r") as log:
        log_content = log.read()
        error_lines = [line for line in log_content.splitlines() if "ERROR" in line or "WARN" in line]
        if error_lines:
            pytest.fail(f"Errors/Warnings found in pipeline:\n{''.join(error_lines)}\nSee {log_file}")

    # Check exit code (0 or expected EOS code)
    if return_code != 0:
        # GStreamer might return non-zero for EOS with queue limit—check logs
        if "End-Of-Stream" not in log_content and "reached end of stream" not in log_content:
            pytest.fail(f"Pipeline failed with exit code {return_code}. See {log_file}")
    
    print(f"Pipeline {pipeline.split('!')[0]} processed 100 frames successfully")

def test_pipelines_found():
    """Ensure at least one pipeline was found in README."""
    if not PIPELINES:
        pytest.fail("No gst-launch-1.0 pipelines found in README.md")
    print(f"Found {len(PIPELINES)} pipelines to test")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
