import pytest
from flask.testing import FlaskClient

from app import app


@pytest.fixture
def client() -> FlaskClient:
    with app.test_client() as client:
        yield client


def test_classify_image(client: FlaskClient) -> None:

    data = {"image": (open("tests/fern.jpg", "rb"), "test.jpg")}

    response = client.post(
        "/classify_image", data=data, content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert 'fern' in response.json[0]['label']
