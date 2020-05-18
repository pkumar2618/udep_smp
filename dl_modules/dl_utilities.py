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

    def update(self, section_name, data, new_file_name=None):
        """
        update will update the data in the configuration.json as well everytime it is called. 
        Note that, only iter_cycle_update creates and save a copy of configuration with new_file_name
        :param section_name: string specifying the sections
        :param data: the data that would go to sections specific settings.
        :return: update the configuration.json file.
        """
        try:
            with open(self.file_str, 'r') as f_read:
                self.config = json.load(f_read)
                for key, value in data.items():
                    self.config[section_name][key] = value
            # note that when configuration is run for the first time we will encounter section_name missing KeyError 
            # therefore this line will not execute. Only in the case of iter_cycle_update this will execute.  
            if new_file_name:
                with open(new_file_name, 'w') as f_write:
                    json.dump(self.config, f_write, indent=4)
            else: # the orginal file should be updated as well with the current info on iteration number and other updates
                with open(self.file_str, 'w') as f_write:
                    json.dump(self.config, f_write, indent=4)

        except KeyError as err_key:
            print(err_key)
            print(f"KeyError: the section {section_name} doesn't exist or is with different name.\n"
                  "creating a new section with the same name")
            with open(self.file_str, 'r') as f_read:
                self.config = json.load(f_read)
                self.config[section_name] = {}
                for key, value in data.items():
                    self.config[section_name][key] = value

            if new_file_name:
                with open(new_file_name, 'w') as f_write:
                    json.dump(self.config, f_write, indent=4)
            else: # the orginal file should be updated as well with the current info on iteration number and other updates
                with open(self.file_str, 'w') as f_write:
                    json.dump(self.config, f_write, indent=4)


    def iter_cycle_update(self):
        try:
            # find if the iteration number exists in the configuraiton file
            iter_no = self.config["training_settings"]["iteration_number"]
            new_file_name = f"{self.file_str}_{iter_no+1}"
            self.update(section_name = "training_settings", data={"iteration_number": iter_no+1})
            self.update(section_name = "training_settings", 
                        data={"iteration_number": iter_no + 1}, 
                        new_file_name=new_file_name)
        except KeyError:
            #add value iteration_number = 0 in the old configuration file if missing.
            self.update(section_name = "training_settings", data={"iteration_number": 1})
            #update iteration number and the name of the configuration file. 
            new_file_name = f"{self.file_str}_{1}"
            self.update(section_name = "training_settings", 
                        data={"iteration_number": 1}, 
                        new_file_name=new_file_name)
    
    def iteration_info(self, iteration_info = None): 
        self.update(section_name = "training_settings", 
                    data={"iteration_info": iteration_info})

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
