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