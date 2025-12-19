# 时间 ： 2020/12/21 9:44
# 作者 ： Dixit
# 文件 ： creat_docker.py
# 项目 ： moziAI_nlz
# 版权 ： 北京华戍防务技术有限公司

import docker
import time

client = docker.from_env()
container = client.containers.create(
     'mozi_innet_v16',
     command='/bin/bash',
     name='pydockertest_%s' % str(6263),
     detach=True,
     tty=True,
     ports={'6060': 6263},
     user='root')
container.start()
out1 = container.exec_run(
    cmd='sh -c "service mysql start && echo success"',
    tty=True,
    user='root',
    detach=False)
print(out1.output)
print('pydockertest_%s mysql is started' % str(6263))
#out2 = container.exec_run(
#    cmd='sh -c "mono /home/LinuxServer/bin/LinuxServer.exe --AiPort 6060"',
#    tty=True,
#    user='root',
#    detach=True)
#print(out2)
out3 = container.exec_run(
    cmd='sh -c "pgrep mono"',
    tty=True,
    user='root',
    detach=False)
if out3.output == b'':
    print('failure!')
print(out3)
