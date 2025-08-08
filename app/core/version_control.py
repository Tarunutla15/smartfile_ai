import subprocess
import os

def run_command(command_list, cwd=None):
    """Run a shell command and raise an exception on failure."""
    result = subprocess.run(command_list, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command '{' '.join(command_list)}' failed:\n{result.stderr}")
    return result.stdout.strip()

def dvc_add_commit_push(path: str, repo_dir: str = None):
    """
    Adds a path to DVC tracking, commits the DVC metadata to Git,
    and pushes the data to the configured DVC remote.

    Args:
        path (str): Path to the file or folder to track.
        repo_dir (str): Directory of your Git/DVC repo. Defaults to current dir.
    """
    repo_dir = repo_dir or os.getcwd()

    try:
        print(f"Running: dvc add {path}")
        run_command(["dvc", "add", path], cwd=repo_dir)

        dvc_file = f"{path}.dvc"
        print(f"Adding {dvc_file} to Git and committing")
        run_command(["git", "add", dvc_file], cwd=repo_dir)
        run_command(["git", "commit", "-m", f"DVC: track {path}"], cwd=repo_dir)

        print("Pushing data to DVC remote...")
        run_command(["dvc", "push"], cwd=repo_dir)

        print(f"DVC tracking and push for {path} completed successfully.")
    except Exception as e:
        print(f"Error during DVC operation: {e}")
        # You can handle exceptions or re-raise if needed
