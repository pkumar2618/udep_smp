import json

class ConfigJSON:
    def __init__(self, file_str):
        self.config = {}
        self.file_str = file_str
        try: # try reading the file if it exists already.
            with open(self.file_str, 'r') as f_read:
                self.config = json.load(f_read)

        except FileNotFoundError:
            with open(self.file_str, 'w') as f_write:
                json.dump(self.config, f_write, indent=4)

    def load(self):
        with open(self.file_str, 'r') as f_read:
            self.config = json.load(f_read)
        # when loading without first instantiating the config object
        # which should never happen as we won't happen the object to call up

    def update(self, section_name, data):
        """

        :param section_name: string specifying the sections
        :param data: the data that would go to sections specific settings.
        :return: update the configuration.json file.
        """
        try:
            with open(self.file_str, 'r') as f_read:
                self.config = json.load(f_read)
                for key, value in data.items():
                    self.config[section_name][key] = value

        except KeyError as err_key:
            print(err_key)
            print(f"KeyError: the section {section_name} doesn't exist or is with different name.\n"
                  "creating a new section with the same name")
            with open(self.file_str, 'r') as f_read:
                self.config = json.load(f_read)
                self.config[section_name] = {}
                for key, value in data.items():
                    self.config[section_name][key] = value

            with open(self.file_str, 'w') as f_write:
                json.dump(self.config, f_write, indent=4)


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


if __name__ == "__main__":
   config = Config(
        testing=True,
        testing_sample=10,
        seed=1,
        batch_size=2,
        lr=3e-4,
        epochs=3,
        hidden_sz=64,
        max_seq_len=100,  # necessary to limit memory usage
        max_vocab_size=100000,
    )
