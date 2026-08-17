"""
Logging and Experiment Tracking setup using Weights & Biases (wandb).
"""

from typing import Any, Dict, Optional


def setup_wandb_logging(project_name: str, config: Dict[str, Any], run_name: Optional[str] = None):
    """
    Stub function to initialize a Weights & Biases (wandb) run.

    Args:
        project_name (str): Name of the W&B project.
        config (dict): Dictionary of model and training hyperparameters.
        run_name (str, optional): Custom name for the run.

    Returns:
        wandb run object when invoked in training scripts, or None stub.
    """
    # In full implementation:
    # import wandb
    # return wandb.init(project=project_name, config=config, name=run_name)
    print(f"[Logging Setup] W&B stub initialized for project '{project_name}'.")
    return None
