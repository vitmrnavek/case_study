# database.py

import psycopg2
import pandas as pd
from google.cloud import bigquery
from pydantic import BaseModel
import time
import yaml
from google.oauth2 import service_account
from case_study.models.metadata import Schema
from dataclasses import dataclass
from typing import Dict, Optional, Union, List
from pathlib import Path

from case_study.utils.logger import get_logger
from case_study.utils.settings import DATABASE_CONFIG_PATH, METADATA_PATH

# Initialize logger for database operations
logger = get_logger("database-queries")


@dataclass
class PostgresConnectionParams:
    host: str
    port: int
    database: str
    user: str
    password: str


@dataclass
class BigQueryConnectionParams:
    project_id: str
    location: str
    credentials_path: str


@dataclass
class ConnectionInfo:
    type: str
    connection_params: Union[PostgresConnectionParams, BigQueryConnectionParams]

    @classmethod
    def from_dict(cls, data: Dict) -> "ConnectionInfo":
        conn_type = data["type"].lower()
        params = data["connection_params"]

        if conn_type == "postgres":
            connection_params = PostgresConnectionParams(**params)
        elif conn_type == "bigquery":
            connection_params = BigQueryConnectionParams(**params)
        else:
            raise ValueError(f"Unsupported connection type: {conn_type}")

        return cls(type=conn_type, connection_params=connection_params)


class ConnectionConfig:
    def __init__(self, config_path: Union[str, Path] = DATABASE_CONFIG_PATH):
        logger.info(f"Initializing ConnectionConfig with path: {config_path}")
        self.config_path = Path(config_path)
        self.connections: Dict[str, ConnectionInfo] = {}
        self.schema_path: Optional[str] = METADATA_PATH
        self._load_config()
        logger.info("ConnectionConfig initialization completed")

    def _load_config(self):
        """Load and validate configuration from YAML file"""
        if not self.config_path.exists():
            logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        logger.info(f"Loading configuration from {self.config_path}")
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
            databases_config = config["databases"]

        # Process each connection
        for conn_name, conn_data in databases_config.items():
            logger.info(f"Processing connection configuration for: {conn_name}")
            try:
                self.connections[conn_name] = ConnectionInfo.from_dict(conn_data)
                logger.info(f"Successfully configured connection: {conn_name}")
            except Exception as e:
                logger.error(
                    f"Invalid configuration for connection '{conn_name}': {str(e)}"
                )
                raise ValueError(
                    f"Invalid configuration for connection '{conn_name}': {str(e)}"
                )

    def get_connection(self, name: str) -> ConnectionInfo:
        """Get connection information by name"""
        logger.info(f"Retrieving connection info for: {name}")
        if name not in self.connections:
            logger.error(f"Connection '{name}' not found in configuration")
            raise KeyError(f"Connection '{name}' not found in configuration")
        return self.connections[name]

    def list_connections(self) -> List[str]:
        """List all available connection names"""
        connections = list(self.connections.keys())
        logger.info(f"Available connections: {connections}")
        return connections

    def get_connection_params(
        self, name: str
    ) -> Union[PostgresConnectionParams, BigQueryConnectionParams]:
        """Get connection parameters for a specific connection"""
        logger.info(f"Retrieving connection parameters for: {name}")
        return self.get_connection(name).connection_params

    def validate(self) -> bool:
        """Validate the entire configuration"""
        logger.info("Validating configuration")
        for conn_name, conn_info in self.connections.items():
            logger.info(f"Validating connection: {conn_name}")
            # Validate PostgreSQL connections
            if conn_info.type == "postgres":
                params = conn_info.connection_params
                if not all(
                    [
                        params.host,
                        params.port,
                        params.database,
                        params.user,
                        params.password,
                    ]
                ):
                    logger.error(
                        f"Invalid PostgreSQL configuration for '{conn_name}': missing required parameters"
                    )
                    raise ValueError(
                        f"Invalid PostgreSQL configuration for '{conn_name}': missing required parameters"
                    )

            # Validate BigQuery connections
            elif conn_info.type == "bigquery":
                params = conn_info.connection_params
                if not all(
                    [params.project_id, params.location, params.credentials_path]
                ):
                    logger.error(
                        f"Invalid BigQuery configuration for '{conn_name}': missing required parameters"
                    )
                    raise ValueError(
                        f"Invalid BigQuery configuration for '{conn_name}': missing required parameters"
                    )

                # Validate credentials file exists
                if not Path(params.credentials_path).exists():
                    logger.error(
                        f"BigQuery credentials file not found: {params.credentials_path}"
                    )
                    raise ValueError(
                        f"BigQuery credentials file not found: {params.credentials_path}"
                    )

            logger.info(f"Connection '{conn_name}' validated successfully")
        logger.info("All connections validated successfully")
        return True

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary format"""
        logger.info("Converting configuration to dictionary format")
        config = {}
        if self.schema_path:
            config["path"] = self.schema_path

        for conn_name, conn_info in self.connections.items():
            config[conn_name] = {
                "type": conn_info.type,
                "connection_params": vars(conn_info.connection_params),
            }
        return config

    def save(self, path: Optional[Union[str, Path]] = None):
        """Save configuration to YAML file"""
        save_path = Path(path) if path else self.config_path
        logger.info(f"Saving configuration to: {save_path}")
        try:
            with open(save_path, "w") as f:
                yaml.dump(self.to_dict(), f)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise


class QueryParams(BaseModel):
    columns: List[str]
    table: str


connections = ConnectionConfig()


class TableIdentifier(BaseModel):

    connection_name: str
    database_type: str
    database_name: str
    schema_name: str = "public"  # Default schema for PostgreSQL
    table_name: str

    def get_duckdb_reference(self) -> str:
        """Get the DuckDB-compatible reference for this table"""
        if self.database_type == "postgres":
            return f"POSTGRES({self.schema_name}.{self.table_name})"
        elif self.database_type == "bigquery":
            return (
                f"BIGQUERY({self.database_name}.{self.schema_name}.{self.table_name})"
            )
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")

    @property
    def full_name(self) -> str:
        """Get the full qualified name of the table"""
        return f"{self.connection_name}.{self.database_name}.{self.schema_name}.{self.table_name}"


class Database:
    def __init__(self, connection_name: str):
        logger.info(f"Initializing database connection for: {connection_name}")
        self.connection_name = connection_name
        self.connection_params = None
        self.config = ConnectionConfig()
        logger.info("Database initialization completed")

    def connect(self, fetch_schema=False):
        """Connect to the database based on the connection type."""
        logger.info(f"Connecting to database: {self.connection_name}")
        try:
            connection_info = self.config.get_connection(self.connection_name)
            self.connection_params = connection_info.connection_params

            if connection_info.type == "postgres":
                logger.info("Establishing PostgreSQL connection")
                self._connect_postgres()
            elif connection_info.type == "bigquery":
                logger.info("Establishing BigQuery connection")
                self._connect_bigquery()
            else:
                logger.error(f"Unsupported connection type: {connection_info.type}")
                raise ValueError(f"Unsupported connection type: {connection_info.type}")

            logger.info("Fetching schema information")
            self.schema = Schema.get_schema(
                self.config.schema_path,
                self.connection_name,
                self.gather_schema,
                fetch_schema,
            )
            logger.info("Database connection established successfully")

        except Exception as e:
            logger.error(f"Failed to connect to {self.connection_name}: {str(e)}")
            raise

    def _connect_postgres(self):
        """Connect to PostgreSQL database"""
        if not isinstance(self.connection_params, PostgresConnectionParams):
            logger.error("Invalid connection parameters for PostgreSQL")
            raise ValueError("Invalid connection parameters for PostgreSQL")

        params = {
            "host": self.connection_params.host,
            "port": self.connection_params.port,
            "database": self.connection_params.database,
            "user": self.connection_params.user,
            "password": self.connection_params.password,
        }

        try:
            self.connection = psycopg2.connect(**params)
            self.cursor = self.connection.cursor()
            logger.info(
                f"Connected to PostgreSQL database: {self.connection_params.database}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise

    def _connect_bigquery(self):
        """Connect to BigQuery database"""
        if not isinstance(self.connection_params, BigQueryConnectionParams):
            logger.error("Invalid connection parameters for BigQuery")
            raise ValueError("Invalid connection parameters for BigQuery")

        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.connection_params.credentials_path
            )

            self.client = bigquery.Client(
                project=self.connection_params.project_id,
                location=self.connection_params.location,
                credentials=credentials,
            )
            logger.info(
                f"Connected to BigQuery project: {self.connection_params.project_id}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to BigQuery: {str(e)}")
            raise

    def gather_schema(self) -> Dict:
        """Gather schema information and metadata statistics for tables and columns"""
        logger.info("Gathering schema information")
        if not self.connection_params:
            logger.error("Connection parameters are not set")
            raise ValueError("Connection parameters are not set.")

        connection_info = self.config.get_connection(self.connection_name)
        try:
            if connection_info.type == "postgres":
                logger.info("Gathering PostgreSQL schema")
                schema_data = self._gather_postgres_schema()
            elif connection_info.type == "bigquery":
                logger.info("Gathering BigQuery schema")
                schema_data = self._gather_bigquery_schema()
            else:
                logger.error(f"Unsupported connection type: {connection_info.type}")
                raise ValueError(f"Unsupported connection type: {connection_info.type}")

            logger.info("Schema gathered successfully")
            return Schema.from_json(schema_data)
        except Exception as e:
            logger.error(f"Failed to gather schema: {str(e)}")
            raise

    def _gather_postgres_schema(self) -> Dict:
        schema_data = {
            "tables": [],
            "database_name": self.connection_params.database,
            "database_type": "postgres",
        }

        # Get all schemas and their tables
        schema_query = """
        SELECT DISTINCT 
            schemaname AS schema_name,
            tablename AS table_name,
            obj_description(format('%s.%s', schemaname, tablename)::regclass::oid) AS table_comment
        FROM pg_catalog.pg_tables
        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY schemaname, tablename;
        """
        self.cursor.execute(schema_query)
        table_metadata = self.cursor.fetchall()

        for schema_name, table_name, description in table_metadata:
            column_metadata = self.get_column_metadata(table_name, schema_name)
            schema_data["tables"].append(
                {
                    "table_name": table_name,
                    "schema_name": schema_name,
                    "table_comment": description,
                    "columns": column_metadata,
                }
            )
        return schema_data

    def get_column_metadata(self, table: str, schema: str = "public") -> List[Dict]:
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
            AND table_schema = '{schema}'
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

    def _gather_bigquery_schema(self) -> Dict:
        schema_data = {
            "database_name": self.connection_params.project_id,
            "tables": [],
            "database_type": "bigquery",
        }

        # Query to get all datasets (schemas) in the project
        datasets = list(self.client.list_datasets())

        for dataset in datasets:
            dataset_id = dataset.dataset_id
            # Get all tables in the dataset
            tables = list(self.client.list_tables(dataset.reference))

            for table in tables:
                table_ref = self.client.get_table(table.reference)

                columns = []
                for field in table_ref.schema:
                    columns.append(
                        {
                            "column_name": field.name,
                            "data_type": field.field_type,
                            "is_nullable": field.is_nullable,
                            "column_comment": field.description or "",
                            "column_default": "",
                        }
                    )

                schema_data["tables"].append(
                    {
                        "table_name": table.table_id,
                        "schema_name": dataset_id,
                        "table_comment": table_ref.description or "",
                        "columns": columns,
                    }
                )

        return schema_data

    def construct_query(self, params: QueryParams) -> str:
        return f"SELECT {', '.join(params.columns)} FROM {params.table}"

    def fetch_data(self, query: str) -> pd.DataFrame:
        pass


class PostgresDatabase(Database):
    def _construct_log_struct(
        self, cursor: psycopg2.extensions.cursor, df: pd.DataFrame, start_time: float
    ) -> dict:
        return {
            "query_text": str(cursor.query),
            "execution_time": time.time() - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)),
            "rows_returned": df.shape[0],
            "user_info": self.connection_params.get("user", "unknown"),
            "transaction_id": self.connection.get_backend_pid(),
        }

    def fetch_data(self, query: str) -> pd.DataFrame:
        try:
            # Validate query with a dry run
            self.connection.autocommit = False
            self.cursor.execute(query)
            self.connection.rollback()  # Rollback to simulate dry run
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error executing query: {e}")
            raise ValueError(f"Query validation failed: {e}")

        start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time))

        # Execute the query for real
        self.cursor.execute(query)
        columns = [desc[0] for desc in self.cursor.description]
        data = self.cursor.fetchall()
        df = pd.DataFrame(data, columns=columns)

        # Log query execution details
        logger.log_struct(self._construct_log_struct(self.cursor, df, start_time))

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

    def get_column_metadata(self, table: str, schema: str = "public") -> List[Dict]:
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
            AND table_schema = '{schema}'
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
    def _construct_log_struct(self, query_job: bigquery.QueryJob, result) -> dict:
        return {
            "query_text": query_job.query,
            "execution_time": query_job.slot_millis,
            "timestamp": query_job.started.strftime("%Y-%m-%d %H:%M:%S"),
            "rows_returned": result.num_results,
            "user_info": "bigquery",  # BigQuery doesn't have user info in the same way
            "transaction_id": query_job.job_id,
        }

    def fetch_data(self, query: str) -> pd.DataFrame:
        # Validate query with a dry run
        job_config = bigquery.QueryJobConfig(dry_run=True)
        try:
            self.client.query(query, job_config=job_config)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise ValueError(f"Query validation failed: {e}")

        # Execute the query for real
        query_job = self.client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Log query execution details
        logger.log_struct(self._construct_log_struct(query_job, result))

        return df

    def get_schema_metadata(self) -> List[Dict]:
        prefix = f"{self.connection_params.project_id}.region-{self.connection_params.location}"
        query = f"""
        SELECT
            table_name,
            column_name,
            data_type,
            is_nullable
        FROM
            `{prefix}.INFORMATION_SCHEMA.COLUMNS`
        ORDER BY
            table_name, ordinal_position
        """

        try:
            query_job = self.client.query(query)
            results = query_job.result()

            tables = {}
            for row in results:
                table_name = row.table_name
                if table_name not in tables:
                    tables[table_name] = {
                        "table_name": table_name,
                        "table_comment": "",  # Default empty comment
                        "rows_count": 0,  # Default row count
                        "columns": [],
                    }

                tables[table_name]["columns"].append(
                    {
                        "column_name": row.column_name,
                        "data_type": row.data_type,
                        "is_nullable": row.is_nullable == "YES",
                        "column_comment": "",  # Default empty comment
                        "column_default": "",  # Default empty default value
                    }
                )

            return list(tables.values())

        except Exception as e:
            logger.error(f"Error fetching BigQuery schema: {str(e)}")
            return []
