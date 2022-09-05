#!/usr/bin/env python3

import sys
import json

from notebook.auth import passwd

value = passwd(sys.argv[1])

config = {}
config["NotebookApp"] = {"password": value}

json.dump(config, sys.stdout, indent=2)
