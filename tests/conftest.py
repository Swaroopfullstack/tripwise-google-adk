import pytest

from portfolio.config import Principal, Settings
from portfolio.runtime import Services


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, app_env="test", data_dir=tmp_path, requests_per_minute=300)


@pytest.fixture
def services(settings):
    runtime = Services(settings)
    yield runtime
    runtime.close()


@pytest.fixture
def alice():
    return Principal("demo", "alice")
