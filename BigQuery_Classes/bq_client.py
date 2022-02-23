from fileinput import filename
from google.cloud import bigquery
from google.oauth2 import service_account

from Variables.create_vars import *

class BQ_Client_Connect:
    def create_client(self):
        credentials = service_account.Credentials.from_service_account_file(filename = gcp_key, 
                                                                            scopes = ["https://www.googleapis.com/auth/cloud-platform"])
        client = bigquery.Client(credentials = credentials, project = gcp_project_id)

        return client