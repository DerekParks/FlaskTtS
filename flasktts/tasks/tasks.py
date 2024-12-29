import json
import os

from huey.signals import SIGNAL_COMPLETE, SIGNAL_ERROR

from flasktts.app import MQTT_TOPIC, huey, mqtt_client
from flasktts.tasks.ffmpeg import convert_wav_to_mp3
from flasktts.tts.style2tts import Style2TTSHighlander


@huey.task(context=True)
@huey.lock_task("gpu-lock")
def style2_tts_task(text: str, task=None):
    """Huey task for Style2TTS text-to-speech conversion.

    Args:
        text (str): Text to convert to speech
        task (Huey task): Huey task object (default: None)

    """

    output_wav = Style2TTSHighlander.get_instance().synth_text(text, task.id)
    output_mp3 = convert_wav_to_mp3(output_wav)

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
