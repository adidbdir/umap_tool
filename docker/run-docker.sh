cd docker
echo "UID=$(id -u)" > .env
echo "GID=$(id -g)" >> .env
docker compose up -d --build
docker compose exec dataset_analyzer bash
cd ..

# cd docker
# # export DOCKER_USER="$(id -u):$(id -g)"
# docker compose up -d
# docker compose exec dataset_analyzer bash
# cd ..