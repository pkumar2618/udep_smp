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
    
    def experiment_info(self, experiment_info = None):
        self.update(section_name = "training_settings", 
                    data={"experiment_info": experiment_info, "iteration_info": None, 
                         "iteration_number": 0})


def combine_input_output(input_file_path, output_file_path):
    f_input = open(input_file_path, 'r')
    json_input = json.load(f_input)

    f_output = open(output_file_path, 'r')
    json_output = json.load(f_output)


    # Note this the following are taken from data_loader read methods. please confirm to that block of code for
    # a meaningful combination of output_data with input_data
    # consider taking spo-triples from next 4 queries in the training set to form negative samples
    # of the current question's spo-triples
    len_input = len(json_input)
    len_output = len(json_output)
    json_temp = []

    for i, question_spos in enumerate(json_input):
        question = question_spos['question']
        for spo_list in question_spos['spos_label']:
            # correct spo: positive sample
            spo_label_joined = ' '.join(spo_list)
            label = 1 # label is 1 for positive sample
            json_temp.append({'question': question, 'spo_triple': spo_label_joined, 'target_score': label})
            # incorrect spo: the negative sample, not that this is not really
            # a batch negative sample, where we pick up spo from the instances
            # in the batch. Which kind of help in speed up, the least we can say.
            for neg_sample in range(4):
                if i < len_input - 4:  # take next 4 queries
                    neg_question_spos = json_input[i + neg_sample + 1]
                else:  # take previous examples
                    neg_question_spos = json_input[i - 1 - neg_sample]
                #         note that we will only take up spo-triples to form the negative examples for training.
                for neg_spo_list in neg_question_spos['spos_label']:
                    neg_spo_label_joined = ' '.join(neg_spo_list)
                    # the question will stay from the positive sample,
                    # only the spo-triple will be taken up for negative example
                    label = 0  # zero label for negative sample
                    json_temp.append({'question': question, 'spo_triple': neg_spo_label_joined, 'target_score': label})

    json_combined = []
    with open("output_input_analysis.json", 'w') as f_write:
        for dict1, dict2 in zip(json_temp, json_output):
            json_combined.append(merge_dict(dict1, dict2))
        json.dump(json_combined, f_write, indent=4)


def merge_dict(dict1, dict2):
    res = {**dict1, **dict2}
    return res

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)



if __name__ == "__main__":
    combine_input_output('../dataset_qald/qald_input.json', 'output_prediction.json')
