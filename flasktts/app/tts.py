from flask import send_file
from flask_restx import Namespace, Resource, fields

from flasktts.app import huey
from flasktts.tasks.tasks import cleanup, style2_tts_task


class JobStatus(fields.String):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"

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
        pending_tasks = set([task.id for task in huey.pending()])

        if job_id in pending_tasks:
            return {"status": JobStatus.PENDING}

        job_results = huey.all_results()
        if job_id in job_results:
            return {"status": JobStatus.COMPLETED}

        api.abort(404, "Job not found")


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

        if job_id not in huey.all_results().keys():
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
        pending_tasks = huey.pending()
        job_results = huey.all_results()

        return {
            "jobs": [
                {
                    "job_id": task.id,
                    "status": JobStatus.PENDING,
                }
                for task in pending_tasks
            ]
            + [
                {
                    "job_id": job_id,
                    "status": JobStatus.COMPLETED,
                }
                for job_id in job_results
            ]
        }

    def delete(self):
        """Delete all text-to-speech jobs"""
        cleanup()

        return "All jobs deleted", 204
