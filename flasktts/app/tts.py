import os
from glob import glob

from flask import send_file
from flask_restx import Namespace, Resource, fields

from flasktts.app import huey
from flasktts.config import Config
from flasktts.tasks.tasks import (
    cleanup,
    get_tasks_pending_failed_complete_running,
    style2_tts_task,
)


class JobStatus(fields.String):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RUNNING = "RUNNING"

    def format(self, value):
        return str(value)


api = Namespace("tts", description="Text-to-Speech conversion endpoints")

# Request model
tts_request = api.model(
    "TTSRequest",
    {
        "text": fields.String(
            required=True,
            description="Text to convert to speech",
            example="Hello world, this is a test of the text to speech system.",
        )
    },
)

# Response models
job_response = api.model(
    "JobResponse",
    {
        "job_id": fields.String(description="Unique job identifier"),
    },
)

job_status = api.model(
    "JobStatus",
    {
        "status": JobStatus(description="Current job status"),
    },
)

jobs_list = api.model(
    "JobsList",
    {
        "jobs": fields.List(
            fields.Nested(
                api.model(
                    "Job",
                    {
                        "job_id": fields.String(description="Job identifier"),
                        "status": JobStatus(description="Job status"),
                    },
                )
            )
        )
    },
)


@api.route("/synthesize")
class TextToSpeechJob(Resource):
    @api.doc(
        "create_tts_job",
        responses={
            202: "Job created successfully",
            400: "Invalid request parameters",
        },
    )
    @api.expect(tts_request)
    @api.marshal_with(job_response)
    def post(self):
        """
        Create a new text-to-speech conversion job

        Returns a job ID that can be used to check status and retrieve the result
        """
        text = api.payload.get("text")
        if not text:
            api.abort(400, "Missing or empty 'text' parameter")

        result = style2_tts_task(text)
        return {"job_id": result.id}, 202


@api.route("/jobs/<string:job_id>")
@api.param("job_id", "The job identifier")
class TextToSpeechStatus(Resource):
    @api.doc(
        "get_job_status", responses={200: "Job status retrieved", 404: "Job not found"}
    )
    @api.marshal_with(job_status)
    def get(self, job_id):
        """Get the status of a text-to-speech job"""
        pending, failed, completed, running = (
            get_tasks_pending_failed_complete_running()
        )

        if job_id in pending:
            return {"status": JobStatus.PENDING}

        if job_id in completed:
            return {"status": JobStatus.COMPLETED}

        if job_id in failed:
            return {"status": JobStatus.FAILED}

        if job_id in running:
            return {"status": JobStatus.RUNNING}

        api.abort(404, "Job not found")

    @api.doc("delete_job", responses={204: "Job deleted successfully", 400: "Error"})
    def delete(self, job_id):
        """Delete a text-to-speech job"""
        pending, failed, completed, running = (
            get_tasks_pending_failed_complete_running()
        )
        if job_id in (failed + completed):
            huey.get(job_id, peek=False)

            for file in glob(f"{Config.STYLE_2_TTS_WORKDIR}/{job_id}.*"):
                os.remove(file)

        elif job_id in pending:
            huey.revoke_by_id(job_id)
        elif job_id in running:
            return "Cancel not supported", 400
        return "Job deleted", 204


@api.route("/jobs/<string:job_id>/download")
@api.param("job_id", "The job identifier")
class TextToSpeechDownload(Resource):
    @api.doc(
        "download_speech",
        responses={
            200: "Audio file retrieved successfully",
            404: "Job not found",
        },
    )
    def get(self, job_id):
        """
        Download the generated audio file

        Only available once the job status is COMPLETED
        """
        _, _, completed, _ = get_tasks_pending_failed_complete_running()
        if job_id not in completed:
            api.abort(404, "Job not found")

        return send_file(
            huey.get(job_id, peek=True),
            mimetype="audio/mpeg",
            as_attachment=True,
            download_name=f"{job_id}.mp3",
        )


@api.route("/jobs")
class TextToSpeechJobs(Resource):
    @api.doc("get_jobs", responses={200: "List of all jobs"})
    @api.marshal_with(jobs_list)
    def get(self):
        """Get a list of all text-to-speech jobs"""
        pending, failed, completed, running = (
            get_tasks_pending_failed_complete_running()
        )
        print(f"Pending2: {pending}")
        print(f"Failed2: {failed}")

        return {
            "jobs": [
                {
                    "job_id": task.id,
                    "status": JobStatus.PENDING,
                }
                for task in pending
            ]
            + [
                {
                    "job_id": job_id,
                    "status": JobStatus.COMPLETED,
                }
                for job_id in completed
            ]
            + [
                {
                    "job_id": job_id,
                    "status": JobStatus.FAILED,
                }
                for job_id in failed
            ]
            + [
                {
                    "job_id": job_id,
                    "status": JobStatus.RUNNING,
                }
                for job_id in running
            ]
        }

    def delete(self):
        """Delete all text-to-speech jobs"""
        cleanup()

        return "All jobs deleted", 204
