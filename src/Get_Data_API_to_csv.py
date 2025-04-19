import requests
import pandas as pd
import logging
import sys
from requests.auth import HTTPBasicAuth
import os

class APICsvExporter:                                   
    """Class to handle API data extraction and CSV conversion with logging"""
    
    def __init__(self, api_url, username, password, csv_filename):
        """
        Initialize the API CSV Exporter
        
        :param api_url: URL of the API endpoint
        :param username: Basic authentication username
        :param password: Basic authentication password
        :param csv_filename: Output CSV filename
        """
        self.api_url = api_url
        self.username = username
        self.password = password
        self.csv_filename = csv_filename
        
        # Configure logging
        logging.basicConfig(
            filename='D:/EUR_AI_MASTER_PROJECTS/AI_Based_Product_Pricing/logs/api_to_csv.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def fetch_data(self):
        """Fetch data from API with Basic Authentication"""
        try:
            self.logger.info(f'Attempting to connect to API: {self.api_url}')
            
            response = requests.get(
                self.api_url,
                auth=HTTPBasicAuth(self.username, self.password),
                timeout=10
            )
            
            response.raise_for_status()
            self.logger.info('Successfully retrieved data from API')
            return response.json()
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f'Request failed: {str(e)}')
            raise

    def convert_to_dataframe(self, api_data):
        """Convert API response to pandas DataFrame"""
        try:
            self.logger.info('Converting API response to DataFrame')
            
            if not api_data:
                self.logger.warning('Empty response from API')
                return pd.DataFrame()
                
            df = pd.DataFrame(api_data['data'])
            self.logger.info(f'DataFrame created with {len(df)} rows and {len(df.columns)} columns')
            return df
        
        except KeyError as e:
            self.logger.error(f'Missing expected key in API response: {str(e)}')
            raise
        except Exception as e:
            self.logger.error(f'DataFrame conversion failed: {str(e)}')
            raise

    def write_to_csv(self, df):
        """Write DataFrame to CSV file"""
        try:
            self.logger.info(f'Writing DataFrame to CSV file: {self.csv_filename}')
            
            if df.empty:
                self.logger.warning('Empty DataFrame - creating empty CSV file')
            
            file_exists = os.path.isfile(self.csv_filename)    
            df.to_csv(self.csv_filename, index=False, encoding='utf-8', mode='a', header= not file_exists)
            self.logger.info(f'Successfully wrote DataFrame to {self.csv_filename}')
        
        except (IOError, PermissionError) as e:
            self.logger.error(f'CSV write failed: {str(e)}')
            raise
        except Exception as e:
            self.logger.error(f'Unexpected error during CSV write: {str(e)}')
            raise

    def run(self):
        """Execute the full workflow"""
        try:
            # Fetch data from API
            api_data = self.fetch_data()
            
            # Convert to DataFrame
            df = self.convert_to_dataframe(api_data)
            
            # Write to CSV
            self.write_to_csv(df)
            
            print(f'Data successfully exported to {self.csv_filename}')
        
        except Exception as e:
            self.logger.critical(f'Process failed: {str(e)}')
            sys.exit(f'Error: {str(e)} (Check logs for details)')


if __name__ == '__main__':
    # Configuration
    config = {
        'api_url': 'https:/basicta',
        'username': 'username',
        'password': 'pw123',
        'csv_filename': 'Product_Pricing_api.csv'
    }
    
    # Create and run exporter
    exporter = APICsvExporter(**config)
    exporter.run()