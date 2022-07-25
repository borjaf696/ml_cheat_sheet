from src.dao.bq import *

class Connector:
    def __init__(self, method = 'big_table'):
        if method == 'big_table':
            self.connect = BQ_access()
    
    def get_data_file(self, file):
        return self.connect.launch_query_from_file(file)

    def get_data_file_parametric(self, file, params = None):
        return self.connect.map_query_parametric(file) if params is None else\
             self.connect.map_query_parametric(file, params)

if __name__ == '__main__':
    cn = Connector()