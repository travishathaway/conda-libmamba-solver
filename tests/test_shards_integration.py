"""
Integration tests for sharded repodata functionality

The integration tests in this module make use of both conda-forge and conda-forge-sharded
channels on anaconda.org
"""
from pathlib import Path

import pytest
from conda.testing.fixtures import TmpEnvFixture

from conda_libmamba_solver.solver import LibMambaSolver
from conda_libmamba_solver.shards_subset import get_channel_shards, get_repodata_subset

#: Contains normal repodata
TEST_CHANNEL = ["conda-forge"]

#: Contains sharded repodata
TEST_CHANNEL_SHARDED = ["conda-forge-sharded"]


@pytest.fixture(autouse=True)
def no_pip(monkeypatch):
    """
    Disable `pip` as a dependency for all of these tests
    """
    monkeypatch.setenv("CONDA_ADD_PIP_AS_PYTHON_DEPENDENCY", "0")


def test_build_repodata_subset_compatibility(tmp_env: TmpEnvFixture, tmp_path: Path):
    """
    Ensure that the repodata subset that's built is compatible with what is fetched
    via the original repodata fetching strategy (i.e. downloading a single `repodata.json` file)
    """
    dependencies = ["requests"]

    with tmp_env("") as prefix:
        solver = LibMambaSolver(
            prefix,
            TEST_CHANNEL,
            specs_to_add=dependencies
        )

        final_state = solver.solve_final_state()
        package_names = [package.name for package in final_state]

        channel_data = get_channel_shards(dependencies, TEST_CHANNEL)
        shards = get_repodata_subset(dependencies, channel_data)

        # `package_names` should be a subset of `shards.nodes.keys()`
        assert set(package_names).difference(shards.nodes.keys()) == set()
