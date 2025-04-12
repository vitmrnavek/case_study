# database.py

import psycopg2
import pandas as pd
from google.cloud import bigquery
from pydantic import BaseModel, Field
from typing import List, Dict
import google.cloud.logging
from google.cloud.logging import Client
import time
import yaml
from google.oauth2 import service_account
from metadata import Schema


# Initialize the Google Cloud Logging client
with open('connections.yaml', 'r') as file:
    connections = yaml.safe_load(file)  
    print("available connections:", list(connections.keys()))

logging_client = Client(project='datamancers',credentials=service_account.Credentials.from_service_account_file(connections["bigquery_datamancers"]["connection_params"]["credentials_path"]))
logger = logging_client.logger('database-queries')


class QueryParams(BaseModel):
    columns: List[str]
    table: str


class Database:
    def __init__(self, connection_name: str):
        self.connection_name = connection_name
        self.connection_params = None

    def connect(self,fetch_schema=False):
        """Connect to the database based on the connection type."""
        if self.connection_name not in connections:
            raise ValueError(f"Connection {self.connection_name} is not available.")
        self.connection_params = connections[self.connection_name]['connection_params']
        connection_type = connections[self.connection_name]['type']
        if connection_type == 'postgres':
            self._connect_postgres()
        elif connection_type == 'bigquery':
            self._connect_bigquery()
        else:
            raise ValueError("Unsupported connection type.")
        if fetch_schema:
            self.schema = self.gather_schema()
            self.schema.save(connections['path'])
        else:
            self.schema = Schema.load(path=connections['path'])
    def create_ai_annotation(self,table_name:str,column_name:str,annotation:str):
        pass
    def _connect_postgres(self):
        self.connection = psycopg2.connect(**self.connection_params)
        self.cursor = self.connection.cursor()
        print("connected to postgres database:", self.connection_params['database'])

    def _connect_bigquery(self):
        credentials = service_account.Credentials.from_service_account_file(
            self.connection_params['credentials_path']
        )
        self.client = bigquery.Client(
            project=self.connection_params['project_id'],
            credentials=credentials
        )
        print("connected to bigquery database:", self.connection_params['project_id'])

    def gather_schema(self) -> Dict:
        """Gather schema information and metadata statistics for tables and columns, structured in a JSON hierarchy."""
        if not self.connection_params:
            raise ValueError("Connection parameters are not set.")
        connection_type = connections[self.connection_name]['type']
        if connection_type == 'postgres':
            schema_data = self._gather_postgres_schema()
        elif connection_type == 'bigquery':
            schema_data = self._gather_bigquery_schema()
        else:
            raise ValueError("Unsupported connection type.")
        return Schema.from_json(schema_data)

    def _gather_postgres_schema(self) -> Dict:
        schema_data = {"tables": [],"database_name": self.connection_params['database'],"database_type": "postgres"}
        schema = self.connection_params.get('schema', 'public')
        table_metadata = self.get_table_metadata(schema)
        for table_name, description in table_metadata:
            column_metadata = self.get_column_metadata(table_name)
            schema_data["tables"].append({"table_name": table_name,
                'table_comment': description,
                'columns': column_metadata
            })
        return schema_data

    def _gather_bigquery_schema(self) -> Dict:
        schema_data = {"database_name": self.connection_params['project_id'],"tables":[],"database_type": "bigquery"}
        
        results = self.get_schema_metadata()
        for result in results:
            schema_data["tables"].append(result)
        return schema_data


    def construct_query(self, params: QueryParams) -> str:
        return f"SELECT {', '.join(params.columns)} FROM {params.table}"

    def fetch_data(self, query: str) -> pd.DataFrame:
        pass

class PostgresDatabase(Database):
    def _construct_log_struct(self, cursor: psycopg2.extensions.cursor,df:pd.DataFrame,start_time:float) -> dict:
        return {
            'query_text': str(cursor.query),
            'execution_time': time.time() - start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time)),
            'rows_returned': df.shape[0],
            'user_info': self.connection_params.get('user', 'unknown'),
            'transaction_id': self.connection.get_backend_pid()
        }
    def fetch_data(self, query: str) -> pd.DataFrame:
        # Log the start time
        
        try:
            # Validate query with a dry run
            self.connection.autocommit = False
            self.cursor.execute(query)
            self.connection.rollback()  # Rollback to simulate dry run
        except Exception as e:
            self.connection.rollback()
            logger.log_text(f"Error executing query: {e}", severity='ERROR')
            raise ValueError(f"Query validation failed: {e}")
        
        start_time = time.time()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time))

        # Execute the query for real
        self.cursor.execute(query)
        columns = [desc[0] for desc in self.cursor.description]
        data = self.cursor.fetchall()
        df = pd.DataFrame(data, columns=columns)


        logger.log_struct(self._construct_log_struct(self.cursor,df,start_time), severity='INFO')

        return df

    def get_table_metadata(self, schema: str) -> List[Dict]:
        query = f"""
        SELECT
            tablename AS table_name,
            obj_description(format('%s.%s', schemaname, tablename)::regclass::oid) AS table_comment
        FROM
            pg_catalog.pg_tables
        WHERE
            schemaname = '{schema}'
        """
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        return data

    def get_column_metadata(self, table: str) -> List[Dict]:
        query = f"""
        SELECT
            column_name,
            data_type,
            is_nullable = 'YES' AS is_nullable,
            column_default,
            col_description(format('%s.%s', table_schema, table_name)::regclass::oid, ordinal_position) AS column_comment
        FROM
            information_schema.columns
        WHERE
            table_name = '{table}'
        """
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        # Convert cursor results to list of dictionaries
        columns = [desc[0] for desc in self.cursor.description]
        result = []
        for row in data:
            row_dict = {}
            for i, value in enumerate(row):
                row_dict[columns[i]] = value
            result.append(row_dict)
        return result

class BigQueryDatabase(Database):
    def _construct_log_struct(self, query_job: bigquery.QueryJob,result) -> dict:
        return {
            'query_text': query_job.query,
            'execution_time': query_job.slot_millis,
            'timestamp': query_job.started.strftime('%Y-%m-%d %H:%M:%S'),
            'rows_returned': result.num_results,
            'user_info': self.connection_params.get('user', 'unknown'),
            'transaction_id': query_job.job_id
        }
    
    def fetch_data(self, query: str) -> pd.DataFrame:

        # Validate query with a dry run
        job_config = bigquery.QueryJobConfig(dry_run=True)
        try:
            self.client.query(query, job_config=job_config)
        except Exception as e:
            logger.log_text(f"Error executing query: {e}", severity='ERROR')
            raise ValueError(f"Query validation failed: {e}")

        # Execute the query for real
        query_job = self.client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Log the query statistics
        logger.log_struct(self._construct_log_struct(query_job,result), severity='INFO')

        return df

    def get_schema_metadata(self) -> pd.DataFrame:
        prefix = f"{self.connection_params['project_id']}.region-{self.connection_params['location']}."
        query = f"""
        WITH tables AS (
        SELECT
            table_name,
            creation_time
        FROM
            `{prefix}INFORMATION_SCHEMA.TABLES`)
        , options AS (
        SELECT
            table_name,
            option_value
        FROM
            `{prefix}INFORMATION_SCHEMA.TABLE_OPTIONS`
            WHERE option_name = 'description'
        ),
        table_size AS (
        SELECT
            table_name,
            total_rows
        FROM
            `{prefix}INFORMATION_SCHEMA.TABLE_STORAGE`
        ),
        join_tables AS (
        SELECT
            tables.table_name,
            tables.creation_time,
            options.option_value AS table_comment,
            table_size.total_rows AS rows_count
        FROM
            tables
        LEFT JOIN options ON tables.table_name = options.table_name
        LEFT JOIN table_size ON tables.table_name = table_size.table_name),
        column_basic AS (
        SELECT
            table_name,
            column_name,
            data_type,
            is_nullable = 'YES' AS is_nullable,
            column_default
        FROM
            `{prefix}.INFORMATION_SCHEMA.COLUMNS`
        ),
        column_details AS (
        SELECT
            table_name,
            column_name,
            description
        FROM
            `{prefix}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
        ), columns_all AS (
        SELECT DISTINCT
            column_basic.table_name,
            column_basic.column_name,
            column_basic.data_type,
            column_basic.is_nullable,
            column_basic.column_default,
            column_details.description AS column_comment
        FROM
            column_basic
        LEFT JOIN
            column_details ON column_basic.column_name = column_details.column_name),

        join_columns AS (
        SELECT
            join_tables.table_name,
            join_tables.creation_time,
            join_tables.table_comment,
            join_tables.rows_count,
            ARRAY_AGG(
            STRUCT(columns_all.column_name, columns_all.column_comment, columns_all.data_type, columns_all.is_nullable, columns_all.column_default)) AS columns
        FROM
            join_tables
            LEFT JOIN 
                columns_all ON join_tables.table_name = columns_all.table_name
        GROUP BY
            1,2,3,4)
        SELECT * FROM join_columns
        """
        query_job = self.client.query(query)
        result = query_job.result()
        result_dict = [dict(i) for i in result]
        return result_dict
