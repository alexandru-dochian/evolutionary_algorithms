import pickle


class DiskUtils:
    @classmethod
    def store(cls, path, content, print_content=True):
        with path.open("wb") as f:
            pickle.dump(content, f)

        if print_content:
            print("{}\n{}\n".format(path, content))

    @classmethod
    def load(cls, path):
        with path.open('rb') as file:
            return pickle.load(file)
