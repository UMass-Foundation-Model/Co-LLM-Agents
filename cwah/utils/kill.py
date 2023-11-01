from psutil import process_iter
from signal import SIGTERM  # or SIGKILL
import sys
def kill(port_numbers):
    for proc in process_iter():
        for conns in proc.connections(kind='inet'):
            if conns.laddr.port in port_numbers:
                proc.send_signal(SIGTERM)  # or SIGKILL

if __name__ == '__main__':
    port_number = sys.argv[1]
    if '-' in port_number:
        port_init = int(port_number.split('-')[0])
        port_end = int(port_number.split('-')[1])
        port_numbers = list(range(port_init, port_end+1))
        kill(port_numbers)

    else:
        kill([int(port_number)])