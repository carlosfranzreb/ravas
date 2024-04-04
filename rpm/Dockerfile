FROM node:lts-alpine3.19

WORKDIR /app

# Install dependencies
COPY tsconfig.json ./
COPY package*.json ./
RUN npm install
RUN npm install -g serve

# # Copy code
COPY ./src/ ./src/
COPY ./public/ ./public/

ENV REACT_APP_WS_URL ws://0.0.0.0:8888
RUN npm run build
# # set environment variables
CMD ["npx","serve", "-s", "build"]
# CMD ["npm", "start"]