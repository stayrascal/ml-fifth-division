#!/usr/bin/env python

import socket, time, thread
socket.setdefaulttimeout(3)

def socket_port(ip, port):
    try:
        if port >= 65535:
            print 'Scaning Port Ending'
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex((ip, port))
        if result == 0:
            lock.acquire()
            print ip, ':', port, 'is been used.'
            lock.release()
    except:
        print 'Scaning Port Exception'

def ip_scan(ip):
    try:

        print 'Starting Scan %s' % ip
        start_time = time.time()
        for i in range(0, 65534):
            thread.start_new_thread(socket_port, (ip, int(i)))
            print 'Finishing Scanning: %.2f' % (time.time() - start_time)
            raw_input('Press Enter to Exit')
    except:
        print 'Scanning Port Exception'

if __name__ == '__main__':
    url = raw_input('Input the IP you want to scan: ')
    lock = thread.allocate_lock()
    ip_scan(url)
