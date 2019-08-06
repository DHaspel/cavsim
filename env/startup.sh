#!/bin/bash

# Install custom python packages at startup
cd /home/jovyan/jupyter/
python setup.py develop

# Start the notepad
cd ~
/usr/local/bin/start-notebook.sh --NotebookApp.password='sha1:5c2c1b5f4bd9:de2d8318c984be9e391a8e1749a007df832c0ffa'
