import zmq
from queue import Queue
from threading import Thread
import time

# maintain available port list
q = Queue(maxsize=200)
port_list = [port for port in range(49353, 49454, 1)]
for port in port_list:
    q.put(port)
# init zmq
zmq_context = zmq.Context()


# docker manager add port resource
def release_port(rcv, que):
    while True:
        port = rcv.recv_pyobj()
        assert type(port) == int
        print(f'release port {port}.')
        que.put(port)


receiver_2 = zmq_context.socket(zmq.PULL)
receiver_2.setsockopt(zmq.RCVHWM, 1)
receiver_2.setsockopt(zmq.SNDHWM, 1)
receiver_2.bind("tcp://*:%s" % '5706')
add_port_thread = Thread(target=release_port, args=(receiver_2, q, ))
add_port_thread.start()
print('release port threading is started.')


# docker manager reply avail port
def reply_port(rcv, que):
    while True:
        msg = rcv.recv_string()
        assert msg == "request port"
        port = que.get()
        print(f'reply port {port}.')
        rcv.send_pyobj(port)


receiver_1 = zmq_context.socket(zmq.REP)
receiver_1.bind("tcp://*:%s" % '5705')
print('reply port threading is started.')
while True:
    msg = receiver_1.recv_string()
    assert msg == "request port"
    port = q.get()
    print(f'reply port {port}.')
    receiver_1.send_pyobj(port)


