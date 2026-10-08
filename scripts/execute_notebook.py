"""Execute and overwrite the portfolio notebook from the repository root."""

import argparse
import os

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from nbformat.v4 import new_output


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "student_dropout_prediction.ipynb"

def execute_in_process(notebook: nbformat.NotebookNode) -> None:
    """Execute cells without a Jupyter socket (useful in restricted sandboxes)."""
    shell = InteractiveShell.instance()
    original_directory = Path.cwd()
    os.chdir(ROOT)
    try:
        execution_count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            execution_count += 1
            with capture_output(display=True) as captured:
                result = shell.run_cell(cell.source)
            if result.error_before_exec:
                raise result.error_before_exec
            if result.error_in_exec:
                raise result.error_in_exec

            outputs = []
            if captured.stdout:
                outputs.append(
                    new_output("stream", name="stdout", text=captured.stdout)
                )
            if captured.stderr:
                outputs.append(
                    new_output("stream", name="stderr", text=captured.stderr)
                )
            outputs.extend(
                new_output(
                    "display_data",
                    data=rich_output.data,
                    metadata=rich_output.metadata,
                )
                for rich_output in captured.outputs
            )
            cell.execution_count = execution_count
            cell.outputs = outputs
    finally:
        os.chdir(original_directory)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--in-process",
    action="store_true",
    help="Execute without starting a Jupyter kernel socket.",
)
args = parser.parse_args()

with NOTEBOOK.open(encoding="utf-8") as handle:
    notebook = nbformat.read(handle, as_version=4)

if args.in_process:
    execute_in_process(notebook)
else:
    client = NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()

with NOTEBOOK.open("w", encoding="utf-8") as handle:
    nbformat.write(notebook, handle)

print(f"Executed {NOTEBOOK}")

