from google.cloud import bigquery
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