#! /usr/bin/python3

import logging
import sys
logging.basicConfig(stream=sys.stderr)

sys.path.insert(0, '/home/daniele.veri.96/flask/flask/')

from server import create_app
application = create_app()

if __name__ == "__main__":
    application.run()
