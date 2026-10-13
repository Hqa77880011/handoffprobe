import multiprocessing as mp
import time


def worker(connection):
    from . import probes, tables

    connection.send(None)
    while True:
        request = connection.recv()
        if request is None:
            break
        module, name, args = request
        started = time.process_time()
        try:
            result = getattr(probes if module == "probes" else tables, name)(*args)
            connection.send((True, result, time.process_time() - started))
        except (
            ValueError,
            KeyError,
            TypeError,
            ZeroDivisionError,
            IndexError,
        ) as error:
            connection.send(
                (
                    False,
                    f"{type(error).__name__}: {error}",
                    time.process_time() - started,
                )
            )
    connection.close()


class Executor:
    def __init__(self, seconds=30, calls=24):
        self.seconds = seconds
        self.calls = calls
        self.ledger = []
        self.process = None

    def __enter__(self):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker, args=(child,), daemon=True)
        self.process.start()
        child.close()
        if not self.connection.poll(60):
            self.close()
            raise RuntimeError("Table executor initialization timed out")
        self.connection.recv()
        return self

    def call(self, module, name, args, timeout=4):
        remaining = self.seconds - sum(row["seconds"] for row in self.ledger)
        if len(self.ledger) >= self.calls or remaining <= 0:
            raise RuntimeError("Executor budget exhausted")
        if self.process is None or not self.process.is_alive():
            raise RuntimeError("Table executor is unavailable after a timeout")
        limit = min(timeout, remaining)
        self.connection.send((module, name, args))
        if not self.connection.poll(limit):
            self.ledger.append({"operation": name, "seconds": limit, "timeout": True})
            self.close()
            raise TimeoutError(f"{name} exceeded its {limit:g}-second reservation")
        success, result, seconds = self.connection.recv()
        self.ledger.append({"operation": name, "seconds": seconds, "timeout": False})
        if not success:
            raise ValueError(result)
        return result

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join()
            self.connection.close()
            self.process = None

    def __exit__(self, *args):
        self.close()
