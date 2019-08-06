# offical docker image from jupyter based on coda
FROM jupyter/base-notebook

USER root

# set working directory to /app
WORKDIR /app

# copy the current directory contents into the container at /app
ADD . /app

# change line endings for startup.sh
RUN sed -i 's/\r//g' /app/startup.sh

# Change directory owner and link python
RUN chown -R $NB_USER:users /app
RUN ln -s /opt/conda/bin/python3 /usr/bin/python3

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_USER
RUN sed -i -e "s/^[#\s]*c\.NotebookApp\.notebook\_dir\s.*$/c\.NotebookApp\.notebook\_dir = '\/home\/jovyan\/jupyter'/g" ~/.jupyter/jupyter_notebook_config.py

# install python packages from requirements
RUN pip install -r requirements.txt

WORKDIR $HOME