#!/usr/bin/env python3

from flasktts.app import create_app
from flasktts.config import Config

app = create_app()

if __name__ == "__main__":
    env = Config.FLASK_ENV
    port = Config.PORT
    if env and env.lower().startswith("dev"):
        app.run(debug=True, host="0.0.0.0", port=port)
    else:
        import waitress

        waitress.serve(app, host="0.0.0.0", port=port)
