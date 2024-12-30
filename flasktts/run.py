#!/usr/bin/env python3

import os

from flasktts.app import create_app

app = create_app()

if __name__ == "__main__":
    env = os.getenv("FLASK_ENV")
    port = os.getenv("PORT", 5001)
    if env and env.lower().startswith("dev"):
        app.run(debug=True, host="0.0.0.0", port=port)
    else:
        import waitress

        waitress.serve(app, host="0.0.0.0", port=port)
