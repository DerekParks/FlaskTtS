import os


class Config:
    """FlaskTtS configuration class"""

    TTS_WORKDIR = os.getenv("STYLE_2_TTS_WORKDIR", "style2tts_workdir")
    HUEY_DB_PATH = os.getenv("HUEY_DB_PATH", "db/huey.db")

    if not os.path.exists(TTS_WORKDIR):
        os.makedirs(TTS_WORKDIR)

    huey_db_dir = os.path.dirname(HUEY_DB_PATH)
    if not os.path.exists(huey_db_dir):
        os.makedirs(huey_db_dir)

    # Get MQTT config from environment
    MQTT_HOST = os.getenv("MQTT_HOST")
    MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
    MQTT_TOPIC = os.getenv("MQTT_TOPIC", "task_events")
    MQTT_USER = os.getenv("MQTT_USER")
    MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

    FLASK_ENV = os.getenv("FLASK_ENV")
    PORT = int(os.getenv("PORT", 5001))

    CLEANUP_TASKS_AFTER_SEC = int(os.getenv("CLEANUP_TASKS_AFTER_SEC", 172800))

    # Qwen3 speech rate: >1.0 = faster, <1.0 = slower, 1.0 = unchanged.
    # Applied via pitch-preserving time stretch after generation.
    QWEN3_SPEECH_RATE = float(os.getenv("QWEN3_SPEECH_RATE", 1.0))

    # Qwen3 chunks generated per batch; halved on CUDA out-of-memory.
    QWEN3_BATCH_SIZE = int(os.getenv("QWEN3_BATCH_SIZE", 4))

    # Kokoro inference under fp16 autocast, CUDA only.
    KOKORO_FP16 = os.getenv("KOKORO_FP16", "true").lower() == "true"

    # Exported model from the homographs repo; unset disables homograph disambiguation.
    HOMOGRAPH_MODEL_DIR = os.getenv("HOMOGRAPH_MODEL_DIR")

    # Wyoming text-to-speech server for Home Assistant, e.g. tcp://0.0.0.0:10200; empty disables it.
    WYOMING_URI = os.getenv("WYOMING_URI", "tcp://0.0.0.0:10200")
