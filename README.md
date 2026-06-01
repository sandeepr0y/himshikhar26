# himshikhar26

### Set Environment
```sh
docker build -t himshikhar26:latest .

docker run \
    --gpus all -d \
    --name ai_ml_container \
    -e DISPLAY=$DISPLAY \
    -v ./:/app \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    himshikhar26:latest
```
