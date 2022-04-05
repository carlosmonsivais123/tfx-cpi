from fileinput import filename
from google.cloud import storage
from google.oauth2 import service_account

from Variables.create_vars import *

class GCS_Client_Connect:
    def create_client(self, project, bucket_name):
        credentials = service_account.Credentials.from_service_account_file(filename = gcp_key, 
                                                                            scopes = ["https://www.googleapis.com/auth/cloud-platform"])

        client = storage.Client(project = project, credentials = credentials)
        bucket = client.get_bucket(bucket_name)

        return bucket