import docker
client = docker.from_env()
# container = client.containers.get('pydockertest')
container_list = client.containers.list(all=True)
for doc in container_list:
    if 'pydockertest' in doc.name:
        container = client.containers.get(doc.name)
        container.stop()
        container.remove()
        print('success remove container: ' + doc.name)
