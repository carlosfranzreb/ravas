
cd ..
docker compose run -v "$(pwd)"/rpm/src:/app/src -v "$(pwd)"/rpm/public:/app/public -v "$(pwd)"/rpm/build:/app/build --entrypoint 'npm run build' --rm rpm
