import time


class Timer:
    def __init__(self):
        self.start_times = {}

    def start(self, name: str):
        self.start_times[name] = time.time()
        print(f"Timer '{name}' started.")

    def get(self,name:str):
        return self.start_times[name]

    def lap(self, name: str):
        if name in self.start_times:
            elapsed_time = time.time() - self.start_times[name]
            print(f"Timer '{name}' stopped. Elapsed time: {elapsed_time:.2f} seconds.")

            return elapsed_time
        else:
            print(f"Timer '{name}' was not started.")