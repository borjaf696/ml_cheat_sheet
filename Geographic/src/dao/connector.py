from src.dao.bq import *

class Connector:
    def __init__(self, method = 'big_table'):
        if method == 'big_table':
            self.connect = BQ_access()
    
    def get_data_file(self, file):
        return self.connect.launch_query_from_file(file)

if __name__ == '__main__':
    cn = Connector()