"""
Logging and Experiment Tracking setup using Weights & Biases (wandb).
"""

import os
from typing import Any, Dict, Optional


def setup_wandb_logging(project_name: str, config: Dict[str, Any], run_name: Optional[str] = None):
    """
    Initialize a Weights & Biases (wandb) run.

    Args:
        project_name (str): Name of the W&B project.
        config (dict): Dictionary of model and training hyperparameters.
        run_name (str, optional): Custom name for the run.

    Returns:
        wandb run object when invoked in training scripts, or None if offline/failed.
    """
    try:
        import wandb
        # Convert ConfigDict or dict to plain dict for JSON serialization
        if hasattr(config, "items"):
            config_dict = dict(config)
        else:
            config_dict = {}

        run = wandb.init(
            project=project_name,
            config=config_dict,
            name=run_name,
            reinit=True,
        )
        print(f"[Logging Setup] W&B run initialized successfully: project='{project_name}', run_name='{run.name if run else run_name}'")
        return run
    except Exception as e:
        print(f"[Logging Setup] Warning: W&B initialization skipped/failed ({e}). Continuing with console logging.")
        return None


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """
    Log metric dictionary to active W&B run if active.
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


def finish_wandb_logging():
    """
    Safely finish active W&B run if active.
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
            print("[Logging Setup] W&B run finished successfully.")
    except Exception:
        pass


