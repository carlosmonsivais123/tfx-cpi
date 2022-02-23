import yaml

class Read_YAML:
    def read_vars(self):
        yaml_file = open("/Users/CarlosMonsivais/Desktop/CPI Project/inputs.yaml")
        parsed_yaml_file = yaml.load(yaml_file, Loader = yaml.FullLoader)

        return parsed_yaml_file