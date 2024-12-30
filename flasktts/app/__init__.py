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
huey = SqliteHuey("tts_tasks", filename=Config.HUEY_DB_PATH)

# Only setup MQTT if host is configured
# This is a temporory fix until I figure out SSEs
mqtt_client = None
if Config.MQTT_HOST:
    mqtt_client = mqtt.Client()
    if Config.MQTT_USER and Config.MQTT_PASSWORD:
        mqtt_client.username_pw_set(Config.MQTT_USER, Config.MQTT_PASSWORD)
    mqtt_client.connect(Config.MQTT_HOST, Config.MQTT_PORT)
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
