## Google Mediapipe Face Tracking in React with Ready Player Me Avatars

This repo is made for this tutorial on youtube: https://youtu.be/GPam8KU3ldw

You can test the live demo at: https://react-face-tracking.vercel.app


## Run

### Local Setup

for local build, install [Node.js][], then run (in this directory)
```bash
npm install
```

### Run Development Service

```bash
npm run start
```


### Docker (Setup & Run Service)

use `docker compose` to start [../docker-compose.yaml](../docker-compose.yaml) (run in project root directory):
```bash
# start all services from compose file
docker compose start

# only start react-app service
docker compose start rpm

# only start react-app service & build/rebuild docker container (and react-app)
docker compose start rpm --build
```

### Docker (Only Build React-App)

if the docker container was not created yet, use `docker compose` with [../docker-compose.yaml](../docker-compose.yaml) (run in project root directory):
```bash
# build docker container only for rpm service (i.e. react-app)
docker compose build rpm
```

then run build script [run_docker_build_rpm.sh](./run_docker_build_rpm.sh) (from this directory):
```bash
# FOR APP: start container to build react-app, then exit container
./run_docker_build_rpm.sh

# FOR WEB EXTENSION: start container to build chrome-extension for react-app, then exit container
./run_docker_build_extension_rpm.sh
```
Afterwards, the compiled built react-app will be in sub-directory `./build`, or for the chrome-extension, in 
sub-directory `./dist`.


For building the packed **chrome-extension** (with signed key, so that a stable extension ID is created), on *nix run
```bash
./run_pack_extension_crx.sh
```

on Windows run
```cmd
run_pack_extension_crx.cmd
```

Afterwards, the created `*.crx` file will be located in `./dist`.


### Use React-App

Then open in browser (recommended: `Google Chrome Browser`):
```
http://localhost:3000
```
_(NOTE the port can be changed by command line arguments)_

For setting a specific _web socket_ address (connecting the the python process), use query parameter `ws`, e.g. for setting
`http://127.0.0.1:8887`:
```
http://localhost:3000?ws=http://127.0.0.1:8887
```

**NOTES:**
 * if you want to change the _web socket_ address, you also need to configure the python process (converter configuration for `Avatar`)
   and, if run from docker container, the port for the `ras` service in (see [../docker-compose.yaml](../docker-compose.yaml))
 * modern browsers will automatically mask characters like `/` within the query parameters,
   when entered into the browser's address field