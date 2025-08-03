from multiprocessing import Queue, Pool

def interate_queue(queue):
    while not queue.empty():
        yield queue.get()

def square(x):
    return x**2

if __name__ == "__main__":
    queue = Queue()

    with Pool() as p:
        for x in p.imap_unordered(square, range(20)):
            queue.put(x)

    # empty queue
    for x in interate_queue(queue):
        print(x)