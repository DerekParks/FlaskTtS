import os


class Config:
    """FlaskTtS configuration class"""

    STYLE_2_TTS_WORKDIR = os.getenv("STYLE_2_TTS_WORKDIR", "style2tts_workdir")
    HUEY_DB_PATH = os.getenv("HUEY_DB_PATH", "huey.db")

    # Get MQTT config from environment
    MQTT_HOST = os.getenv("MQTT_HOST")
    MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
    MQTT_TOPIC = os.getenv("MQTT_TOPIC", "task_events")
    MQTT_USER = os.getenv("MQTT_USER")
    MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

    FLASK_ENV = os.getenv("FLASK_ENV")
    PORT = int(os.getenv("PORT", 5001))

    CLEANUP_TASKS_AFTER_SEC = int(os.getenv("CLEANUP_TASKS_AFTER_SEC", 172800))
