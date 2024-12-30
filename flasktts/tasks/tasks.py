import json
import os

from huey.signals import SIGNAL_COMPLETE, SIGNAL_ERROR
from huey.utils import Error

from flasktts.app import MQTT_TOPIC, huey, mqtt_client
from flasktts.tasks.ffmpeg import convert_wav_to_mp3
from flasktts.tts.style2tts import Style2TTSHighlander


def get_tasks_pending_failed_complete_running() -> (
    tuple[list[str], list[str], list[str], list[str]]
):
    """Get all tasks in the Huey task queue.

    Returns:
        tuple[list[str], list[str], list[str], list[str]]: Tuple of lists of pending, failed, completed, and running tasks

    """

    pending_tasks = huey.pending()
    found_keys = huey.all_results().keys()

    failed = []
    completed = []
    running = []
    for id in found_keys:
        result = huey.get(id, peek=True)
        if isinstance(result, Error):
            failed.append(id)
        elif isinstance(result, str) and "gpu-lock" not in id:
            completed.append(id)
        elif id == "gpu-lock-running":
            running.append(result)

    return pending_tasks, failed, completed, running


@huey.on_startup()
def startup():
    """Huey startup function. Revokes any failed or running tasks and flushes the GPU lock."""
    huey.flush_locks("gpu-lock")
    _, failed, _, running = get_tasks_pending_failed_complete_running()
    for task_id in failed + running:
        huey.revoke_by_id(task_id)
        huey.get(task_id, peek=False)
    huey.get("gpu-lock-running", peek=False)


@huey.task(context=True)
@huey.lock_task("gpu-lock")
def style2_tts_task(text: str, task=None):
    """Huey task for Style2TTS text-to-speech conversion.

    Args:
        text (str): Text to convert to speech
        task (Huey task): Huey task object (default: None)

    """
    huey.put("gpu-lock-running", task.id)

    output_wav = Style2TTSHighlander.get_instance().synth_text(text, task.id)
    output_mp3 = convert_wav_to_mp3(output_wav)

    huey.get("gpu-lock-running", peek=False)
    return os.path.abspath(output_mp3)


@huey.task()
def cleanup():
    """Huey task to clean up old task results."""
    Style2TTSHighlander.get_instance().cleanup()
    huey.flush()


@huey.signal(SIGNAL_COMPLETE)
def task_complete(signal, task):
    print(f"Task {task.id} completed")
    if mqtt_client:
        message = json.dumps({"type": "complete", "task_id": task.id})
        mqtt_client.publish(MQTT_TOPIC, message)


@huey.signal(SIGNAL_ERROR)
def task_error(signal, task):
    print(f"Task {task.id} failed")
    if mqtt_client:
        message = json.dumps({"type": "error", "task_id": task.id, "error": str(exc)})
        mqtt_client.publish(MQTT_TOPIC, message)


if __name__ == "__main__":
    print(f"Number of huey tasks: {len(huey)}")
    huey.results()
