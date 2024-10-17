
cd ..
docker compose run -v "$(pwd)"/rpm/src:/app/src -v "$(pwd)"/rpm/public:/app/public -v "$(pwd)"/rpm/build:/app/build -v "$(pwd)"/rpm/resources:/app/resources "$(pwd)"/rpm/dist:/app/dist --entrypoint 'npm run build:extension' --rm rpm
