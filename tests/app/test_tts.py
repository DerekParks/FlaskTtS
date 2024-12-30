from unittest.mock import MagicMock, patch

import pytest
from flask import Response
from huey.api import Result


def create_mock_result(mock_huey, job_id="test-job-id"):
    task = MagicMock()
    task.id = job_id
    result = Result(mock_huey, task)
    return result


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def mock_huey():
    with patch("flasktts.tasks.tasks.huey") as mock:
        mock.pending.return_value = []
        mock.all_results.return_value = {}
        yield mock


class TestTextToSpeechJob:
    def test_create_tts_job_success(self, client, mock_huey):
        # Arrange
        mock_result = create_mock_result(mock_huey)
        with patch("flasktts.app.tts.style2_tts_task", return_value=mock_result):
            # Act
            response = client.post("/tts/synthesize", json={"text": "Test text"})

            # Assert
            assert response.status_code == 202
            assert response.json == {"job_id": "test-job-id"}

    def test_create_tts_job_missing_text(self, client):
        # Act
        response = client.post("/tts/synthesize", json={})

        # Assert
        assert response.status_code == 400


class TestTextToSpeechStatus:
    def test_get_job_status_pending(self, client, mock_huey):
        # Arrange
        job_id = "pending-job"
        mock_huey.pending.return_value = [job_id]
        # Act
        response = client.get(f"/tts/jobs/{job_id}")

        # Assert
        assert response.status_code == 200
        assert response.json == {"status": "PENDING"}

    def test_get_job_status_completed(self, client, mock_huey):
        # Arrange
        job_id = "completed-job"
        mock_huey.all_results.return_value = {job_id: "result"}
        mock_huey.get.return_value = "result"
        # Act
        response = client.get(f"/tts/jobs/{job_id}")

        # Assert
        assert response.status_code == 200
        assert response.json == {"status": "COMPLETED"}

    def test_get_job_status_not_found(self, client, mock_huey):
        # Act
        response = client.get("/tts/jobs/nonexistent-job")

        # Assert
        assert response.status_code == 404


class TestTextToSpeechDownload:
    def test_download_completed_job(self, client, mock_huey):
        # Arrange
        job_id = "completed-job"
        mock_audio = "fake audio data"
        mock_huey.all_results.return_value = {job_id: mock_audio}
        mock_huey.get.return_value = mock_audio

        with patch("flasktts.app.tts.send_file") as mock_send_file:
            mock_send_file.return_value = Response(
                "fake audio data", content_type="audio/mpeg"
            )
            # Act
            response = client.get(f"/tts/jobs/{job_id}/download")

        # Assert
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "audio/mpeg"

    def test_download_nonexistent_job(self, client, mock_huey):
        # Act
        response = client.get("/tts/jobs/nonexistent-job/download")

        # Assert
        assert response.status_code == 404


class TestTextToSpeechJobs:
    def test_get_all_jobs(self, client, mock_huey):
        # Arrange
        pending_task = MagicMock()
        pending_task.id = "pending-job"
        mock_huey.pending.return_value = [pending_task]
        mock_huey.all_results.return_value = {"completed-job": "result"}
        mock_huey.get.return_value = "result"

        # Act
        response = client.get("/tts/jobs")

        # Assert
        assert response.status_code == 200
        assert response.json == {
            "jobs": [
                {"job_id": "pending-job", "status": "PENDING"},
                {"job_id": "completed-job", "status": "COMPLETED"},
            ]
        }

    def test_delete_all_jobs(self, client):
        # Arrange
        with patch("flasktts.app.tts.cleanup") as mock_cleanup:
            # Act
            response = client.delete("/tts/jobs")

            # Assert
            assert response.status_code == 204
            mock_cleanup.assert_called_once()
