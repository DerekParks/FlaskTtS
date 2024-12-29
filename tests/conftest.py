import pytest

from flasktts.app import create_app


@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    return app
