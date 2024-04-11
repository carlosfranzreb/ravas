FROM node:lts-alpine3.19

WORKDIR /app

# Install dependencies
COPY tsconfig.json ./
COPY package*.json ./
RUN npm install
RUN npm install -g serve

# Copy code
COPY ./src/ ./src/
COPY ./public/ ./public/

# Define env variable
ARG REACT_APP_WS_URL
ENV REACT_APP_WS_URL=$REACT_APP_WS_URL

RUN npm run build
CMD ["npx","serve", "-s", "build"]