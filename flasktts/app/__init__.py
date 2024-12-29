import os

import paho.mqtt.client as mqtt
from flask import Flask
from flask_restx import Api
from huey import SqliteHuey

from flasktts.config import Config

# Initialize API
api = Api(
    title="TTS API", version="1.0", description="Text to Speech API with job queuing"
)

# Initialize Huey with SQLite storage
huey = SqliteHuey("tts_tasks", filename="huey.db")

# Get MQTT config from environment
MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "task_events")
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

# Only setup MQTT if host is configured
mqtt_client = None
if MQTT_HOST:
    mqtt_client = mqtt.Client()
    if MQTT_USER and MQTT_PASSWORD:
        mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    mqtt_client.connect(MQTT_HOST, MQTT_PORT)
    mqtt_client.loop_start()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    api.init_app(app)

    # Register namespaces
    from flasktts.app.health import api as health_ns
    from flasktts.app.tts import api as tts_ns

    api.add_namespace(health_ns)
    api.add_namespace(tts_ns)

    return app
