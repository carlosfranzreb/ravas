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
COPY .env .env

RUN npm run build
CMD ["npx","serve", "-s", "build"]