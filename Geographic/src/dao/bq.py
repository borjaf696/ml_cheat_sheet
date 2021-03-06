from google.cloud import bigquery
import os
class BQ_access:
    def __init__(self):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/temporaryadmin/.config/gcloud/legacy_credentials/borja.freire@deliveryhero.com/adc.json"
        os.environ['GOOGLE_CLOUD_PROJECT'] = "logistics-data-staging-flat"
        self.bq_client = bigquery.Client()

    def __read_query(self, f):
        query = ''
        with open(f,'r+') as file_query:
            lines = file_query.readlines()
            for line in lines:
                query += line.strip()+' '
        return query

    def launch_query_from_file(self, file):
        return self.execute_bq(self.__read_query(file))

    def execute_bq(self, query, output_type = 'df'):
        if output_type == 'df':
            return self.bq_client.query(query).result().to_dataframe()

    def map_query_parametric(self, file_read,  parameters = {'$1':"'qa'", '$2':"'2022-06-01'", '$3':"'2022-07-01'", '$4':'1000000'}):
        '''
        TODO: End
        '''
        query = self.__read_query(file_read)
        for key, val in parameters.items():
            query = query.replace(key,val)
        print('SQL query: ',query)
        return self.execute_bq(query)
