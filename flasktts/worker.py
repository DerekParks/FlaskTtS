"""Huey consumer and Wyoming TTS server in one process, sharing one Kokoro instance."""

import asyncio
import logging
import os
import threading

from huey.consumer_options import ConsumerConfig

from flasktts.config import Config
from flasktts.tasks.tasks import huey
from flasktts.tts.kokorotts import KokoroTTSHighlander

_LOGGER = logging.getLogger(__name__)


def run_wyoming(uri: str) -> None:
    from flasktts.wyoming_server import kokoro_voices, serve

    try:
        asyncio.run(serve(uri, KokoroTTSHighlander.get_instance(), kokoro_voices()))
    except BaseException:
        _LOGGER.exception("Wyoming server stopped")
    os._exit(1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s:%(threadName)s:%(message)s")
    if Config.WYOMING_URI:
        from flasktts.wyoming_server import LANGUAGE_DEFAULT_VOICES

        KokoroTTSHighlander.get_instance().warm_up(LANGUAGE_DEFAULT_VOICES.values())
        threading.Thread(target=run_wyoming, args=(Config.WYOMING_URI,), name="wyoming", daemon=True).start()

    config = ConsumerConfig(workers=1)
    config.validate()
    huey_logger = logging.getLogger("huey")
    config.setup_logger(huey_logger)
    huey_logger.propagate = False
    huey.create_consumer(**config.values).run()


if __name__ == "__main__":
    main()
