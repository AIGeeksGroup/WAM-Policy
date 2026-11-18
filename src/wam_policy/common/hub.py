"""Hugging Face repository id of the QLM-Bench dataset (site configuration)."""
from wam_policy.common import site


def dataset_repo_id(value=None):
    return value or site.get("QLM_DATASET_REPO")
