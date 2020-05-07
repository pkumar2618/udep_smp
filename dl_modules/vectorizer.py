# Loading config_settings
from configparser import ConfigParser
config = ConfigParser()
config.read('bert_clf_config.ini')



# apparently we need to truncate the sequence here, which is a stupid design decision