# Dockerfiles

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


