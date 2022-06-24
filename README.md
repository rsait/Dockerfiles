# Dockerfiles

Repository for some dockerfiles.

# Table of Contents
1. [Workflow example](#Workflow-example)
2. [Silence Speaks](#Silence-Speaks)
3. [Dataprep](#Dataprep)

## Workflow example

This is an example of a workflow with `docker`.

Once you have `docker` installed, you have to create a dockerfile to create an image and then run the container, and sometimes it is difficult to remember all the options.

Therefore in this repository there is an example that could be useful for your own work.

1. Create the dockerfile, which is the blueprint for building the container. Here you have the file `keras.docker`. Write down the instructions to build the image and run the container as comments in the header of the file. USER_ID and GROUP_ID are needed to assure that if you modify or create files when inside the container, those files have the right permissions.

2. Build the image with the command you should write down in the header:

```bash
docker build -f keras.docker -t keras --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
```

3. Run the container to make sure that everything is ok. This command should also be in the header:

```bash
docker run -it --rm -e DISPLAY=unix$DISPLAY -v /home/bee:/tmp -v /tmp/.X11-unix:/tmp/.X11-unix keras /bin/bash -c 'cat /keras.readme.txt; bash'
```

In `keras.readme.txt` you should put info that you think useful for helping you with your project: useful commands, what you are doing, where are the files you need, deadlines, etc.

4. If everything is ok, create and alias for the docker run command and add it as the last line in your `.bashrc` file, so it is available when you log in:

```bash
alias run_keras="docker run -it --rm -e DISPLAY=unix$DISPLAY -v /home/bee:/tmp -v /tmp/.X11-unix:/tmp/.X11-unix keras /bin/bash -c 'cat /keras.readme.txt; bash'"
```

5. And now you can just execute `run_keras` and your container is working after displaying the info in `keras.readme.txt`.

## Silence Speaks

The dockerfile `silencespeaks.docker` encapsulates a web app to test the configuration recognizer demo.

To build the image, execute 

`docker build -f silencespeaks.docker -t silencespeaks .` 

when located in the main directory of this repository.

To run the container

`docker run -it --rm -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -p 8050:8050 --device /dev/video0 silencespeaks /bin/bash -c 'cat /silencespeaks.readme.txt; cd /silencespeaks; python3 app.py'`

The web server starts and to connect to it go to `http://0.0.0.0:8050/` in a tab in your browser.

Then do:

1. Click EXPERT
2. Click TRAIN CONFIGURATIONS
3. Select Medoids

## Dataprep

The dockerfile `dataprep.docker` creates a container to work with the `dataprep` package, a Python package to create reports in html format.

To see an example with the Titanic data, run:

```bash
python3 /dataprep/dataprep_titanic_example.py
```

The results are saved in `/tmp/report_titanic.html`

To see an example of working with two-dimensional numpy arrays, run:

```bash
python3 /dataprep/dataprep_numpy_example.py
```

The results are saved in `/tmp/report_random.html`

