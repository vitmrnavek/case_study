import duckdb
import yaml
from google.cloud import bigquery
from google.api_core import exceptions as google_exceptions
import pandas as pd
from typing import Tuple, Dict
from datetime import datetime
import json
from google.oauth2 import service_account
from case_study.integrations.database import ConnectionConfig
from case_study.utils.settings import DUCKDB_CONFIG
from case_study.utils.logger import get_logger
import os

# Initialize logger for DuckDB operations
logger = get_logger("duckdb-integrator")


class DuckDBIntegrator:
    def __init__(self, config_path=DUCKDB_CONFIG["config_path"]):
        """Initialize the DuckDB integrator with configuration.

        Args:
            config_path: Path to the configuration file
        """
        logger.info(f"Initializing DuckDB integrator with config path: {config_path}")

        # Use the configured database path for permanent storage
        db_path = (
            DUCKDB_CONFIG["database_path"]
            if not DUCKDB_CONFIG["in_memory"]
            else ":memory:"
        )
        logger.info(f"Using DuckDB database path: {db_path}")

        # Create directory for database file if it doesn't exist
        if db_path != ":memory:":
            db_dir = os.path.dirname(db_path)
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Ensured database directory exists: {db_dir}")

        self.duck_conn = duckdb.connect(db_path)
        logger.info("DuckDB connection established")

        self.config = ConnectionConfig(config_path)
        logger.info("Configuration loaded")

        # Setup connections
        self._setup_connections()

        # Create performance logging table
        self._setup_logging_table()
        logger.info("DuckDB integrator initialization completed")

    def _load_config(self, config_path):
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            dict: Loaded configuration data

        Raises:
            Exception: If configuration loading fails
        """
        logger.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                logger.info("Configuration loaded successfully")
                return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise

    def _setup_logging_table(self):
        """Create table for storing query performance logs.

        Raises:
            Exception: If table creation fails
        """
        logger.info("Setting up query logging table")
        try:
            self.duck_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_logs (
                    query_id BIGINT,
                    timestamp TIMESTAMP,
                    source VARCHAR,  -- 'postgres', 'bigquery', or 'duckdb'
                    query_text VARCHAR,
                    execution_time_ms DOUBLE,
                    rows_affected INTEGER,
                    execution_plan JSON,
                    error VARCHAR
                );
            """
            )
            logger.info("Query logging table created successfully")
        except Exception as e:
            logger.error(f"Failed to create logging table: {str(e)}")
            raise

    def _log_query_performance(
        self, source: str, query: str, stats: Dict, error: str = None
    ):
        """Log query performance metrics.

        Args:
            source: Source database type ('postgres', 'bigquery', or 'duckdb')
            query: SQL query that was executed
            stats: Dictionary containing performance statistics
            error: Optional error message if query failed
        """
        logger.info(f"Logging query performance for source: {source}")
        try:
            self.duck_conn.execute(
                """
                INSERT INTO query_logs 
                (query_id, timestamp, source, query_text, execution_time_ms, rows_affected, execution_plan, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
                (
                    abs(hash(f"{query}{datetime.now()}")),  # unique query_id
                    datetime.now(),
                    source,
                    query,
                    stats.get("execution_time_ms", 0),
                    stats.get("rows_affected", 0),
                    json.dumps(stats.get("execution_plan", {})),
                    error,
                ),
            )
            logger.info("Query performance logged successfully")
        except Exception as e:
            logger.error(f"Failed to log query performance: {str(e)}")

    def _setup_connections(self):
        """Setup all database connections.

        Sets up PostgreSQL and BigQuery connections based on configuration.
        Installs and loads necessary extensions.

        Raises:
            Exception: If connection setup fails
        """
        logger.info("Setting up database connections")

        # Install and load PostgreSQL extension
        try:
            logger.info("Installing and loading PostgreSQL extension")
            self.duck_conn.install_extension("postgres")
            self.duck_conn.load_extension("postgres")
            logger.info("PostgreSQL extension loaded successfully")
        except Exception as e:
            logger.error(f"Failed to setup PostgreSQL extension: {str(e)}")
            raise

        # Setup PostgreSQL connections
        postgres_count = 0
        bigquery_count = 0

        for conn_name, conn_info in self.config.connections.items():
            if conn_info.type == "postgres":
                logger.info(f"Setting up PostgreSQL connection: {conn_name}")
                params = conn_info.connection_params
                connection_string = (
                    f"host={params.host} "
                    f"port={params.port} "
                    f"dbname={params.database} "
                    f"user={params.user} "
                    f"password={params.password}"
                )
                try:
                    self.duck_conn.execute(
                        f"""
                        INSTALL postgres_scanner;
                        LOAD postgres_scanner;
                        ATTACH 'postgres:{connection_string}' AS postgres_db;
                    """
                    )
                    postgres_count += 1
                    logger.info(f"PostgreSQL connection {conn_name} setup successfully")
                except Exception as e:
                    logger.error(
                        f"Failed to setup PostgreSQL connection {conn_name}: {str(e)}"
                    )

            elif conn_info.type == "bigquery":
                logger.info(f"Setting up BigQuery connection: {conn_name}")
                params = conn_info.connection_params
                try:
                    self.bq_client = bigquery.Client(
                        project=params.project_id,
                        location=params.location,
                        credentials=service_account.Credentials.from_service_account_file(
                            params.credentials_path
                        ),
                    )
                    bigquery_count += 1
                    logger.info(f"BigQuery connection {conn_name} setup successfully")
                except Exception as e:
                    print(
                        f"Warning: Failed to setup BigQuery connection {conn_name}: {str(e)}"
                    )

    def get_query_logs(
        self, start_time: datetime = None, end_time: datetime = None
    ) -> pd.DataFrame:
        """Retrieve query performance logs within the specified time range.

        Args:
            start_time: Optional start time to filter logs
            end_time: Optional end time to filter logs

        Returns:
            pd.DataFrame: Query logs including performance metrics

        Raises:
            Exception: If log retrieval fails
        """
        logger.info(
            f"Retrieving query logs (start_time={start_time}, end_time={end_time})"
        )

        query = """
            SELECT *
            FROM query_logs
            WHERE 1=1
        """

        if start_time:
            query += f" AND timestamp >= '{start_time}'"
        if end_time:
            query += f" AND timestamp <= '{end_time}'"

        query += " ORDER BY timestamp DESC"

        try:
            logs = self.duck_conn.execute(query).fetchdf()
            logger.info(f"Retrieved {len(logs)} query log entries")
            return logs
        except Exception as e:
            logger.error(f"Failed to retrieve query logs: {str(e)}")
            raise

    def _validate_bigquery_table(self, project: str, dataset: str, table: str) -> bool:
        """Validate that a BigQuery table exists and is accessible.

        Args:
            project: BigQuery project ID
            dataset: BigQuery dataset name
            table: BigQuery table name

        Returns:
            bool: True if table exists and is accessible

        Raises:
            ValueError: If table doesn't exist or isn't accessible
        """
        try:
            table_ref = f"{project}.{dataset}.{table}"
            self.bq_client.get_table(table_ref)
            return True
        except google_exceptions.NotFound:
            raise ValueError(f"BigQuery table '{table_ref}' does not exist")
        except google_exceptions.Forbidden:
            raise ValueError(f"No access to BigQuery table '{table_ref}'")
        except Exception as e:
            raise ValueError(f"Error validating BigQuery table '{table_ref}': {str(e)}")

    def _validate_postgres_table(self, schema: str, table: str) -> bool:
        """Validate that a PostgreSQL table exists and is accessible.

        Args:
            schema: PostgreSQL schema name
            table: PostgreSQL table name

        Returns:
            bool: True if table exists and is accessible

        Raises:
            ValueError: If table doesn't exist or isn't accessible
        """
        try:
            # Check if table exists and is accessible
            result = self.duck_conn.execute(
                f"""
                SELECT EXISTS (
                    SELECT 1 
                    FROM postgres_db.information_schema.tables 
                    WHERE table_schema = '{schema}' 
                    AND table_name = '{table}'
                );
            """
            ).fetchone()[0]

            if not result:
                raise ValueError(f"PostgreSQL table '{schema}.{table}' does not exist")

            # Try to select 0 rows to verify permissions
            self.duck_conn.execute(
                f"SELECT * FROM postgres_db.{schema}.{table} LIMIT 0"
            )
            return True
        except duckdb.Error as e:
            raise ValueError(
                f"Error accessing PostgreSQL table '{schema}.{table}': {str(e)}"
            )

    def _parse_table_reference(self, ref: str, source: str) -> Tuple[str, ...]:
        """Parse a table reference string into its components.

        Args:
            ref: Table reference string (e.g. 'project.dataset.table' for BigQuery)
            source: Source database type ('postgres' or 'bigquery')

        Returns:
            Tuple[str, ...]: Components of the table reference

        Raises:
            ValueError: If reference format is invalid
        """
        parts = ref.split(".")

        if source == "bigquery":
            if len(parts) != 3:
                raise ValueError(
                    "BigQuery table reference must be in format: 'project.dataset.table'"
                )
            return tuple(parts)  # project, dataset, table

        elif source == "postgres":
            if len(parts) != 2:
                raise ValueError(
                    "PostgreSQL table reference must be in format: 'schema.table'"
                )
            return tuple(parts)  # schema, table

        raise ValueError(f"Unknown source type: {source}")

    def run_query(self, query):
        """
        Execute an analysis query that can reference both BigQuery and PostgreSQL.
        The query should use the following special syntax:
        - BIGQUERY('project.dataset.table') to reference BigQuery tables
        - POSTGRES('schema.table') to reference PostgreSQL tables
        Example:
        WITH
            bq_data AS (
                SELECT customer_id, name, email
                FROM BIGQUERY('your-project.your_dataset.customers')
            ),
            pg_data AS (
                SELECT customer_id, order_id, value
                FROM POSTGRES('public.orders')
            ),
            combined AS (
                SELECT b.*, p.*
                FROM bq_data b
                JOIN pg_data p ON b.id = p.id
            )
        SELECT * FROM combined;

        Args:
            query: SQL query to execute

        Returns:
            pd.DataFrame: Query results

        Raises:
            Exception: If query execution fails
        """
        start_time = datetime.now()

        try:
            # First, find and load all BigQuery tables
            while "BIGQUERY(" in query:
                start = query.find("BIGQUERY(")
                end = self._find_matching_parenthesis(query, start + 8)
                if end == -1:
                    raise ValueError("Malformed BIGQUERY() call")

                table_ref = query[start + 9 : end].strip("'")  # Extract table reference

                # Parse and validate table reference format
                project, dataset, table = self._parse_table_reference(
                    table_ref, "bigquery"
                )

                # Validate table exists and is accessible
                self._validate_bigquery_table(project, dataset, table)

                # Load the entire table into a temporary DuckDB table
                temp_table_name = f"bq_temp_{abs(hash(table_ref))}"
                table_id = f"`{table_ref}`"

                # Execute BigQuery with job statistics
                bq_job = self.bq_client.query(f"SELECT * FROM {table_id}")
                df = bq_job.to_dataframe()
                # Convert BigQuery DATE columns to datetime to ensure compatibility with DuckDB
                for col in df.columns:
                    if pd.api.types.is_dtype_equal(df[col].dtype, "dbdate"):
                        df[col] = pd.to_datetime(df[col])

                # Log BigQuery performance
                self._log_query_performance(
                    "bigquery",
                    f"SELECT * FROM {table_id}",
                    {
                        "execution_time_ms": (
                            bq_job.ended - bq_job.started
                        ).total_seconds()
                        * 1000,
                        "rows_affected": len(df),
                        "execution_plan": {
                            "total_bytes_processed": bq_job.total_bytes_processed,
                            "total_bytes_billed": bq_job.total_bytes_billed,
                        },
                    },
                )

                self.duck_conn.register(temp_table_name, df)
                query = query[:start] + temp_table_name + query[end + 1 :]

            # Handle PostgreSQL table references
            while "POSTGRES(" in query:
                start = query.find("POSTGRES(")
                end = self._find_matching_parenthesis(query, start + 8)
                if end == -1:
                    raise ValueError("Malformed POSTGRES() call")

                table_ref = query[start + 9 : end].strip("'")  # Extract table reference
                schema, table = self._parse_table_reference(table_ref, "postgres")
                self._validate_postgres_table(schema, table)

                # Get PostgreSQL execution plan and metrics
                # First get the actual row count
                count_query = f"SELECT COUNT(*) FROM postgres_db.{schema}.{table}"
                row_count = self.duck_conn.execute(count_query).fetchone()[0]

                # Then get the execution plan and timing
                explain_query = f"""
                EXPLAIN (ANALYZE, FORMAT JSON) 
                SELECT * FROM postgres_db.{schema}.{table}
                """
                start_time = datetime.now()
                explain_result = self.duck_conn.execute(explain_query).fetchone()[1]
                execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

                # Parse the execution plan
                try:
                    if isinstance(explain_result, str):
                        plan = json.loads(explain_result)
                    else:
                        plan = explain_result

                    # Log PostgreSQL performance
                    self._log_query_performance(
                        "postgres",
                        f"SELECT * FROM {schema}.{table}",
                        {
                            "execution_time_ms": execution_time_ms,
                            "rows_affected": row_count,
                            "execution_plan": plan,
                        },
                    )
                except Exception as e:
                    logger.error(f"Error parsing execution plan: {str(e)}")
                    self._log_query_performance(
                        "postgres",
                        f"SELECT * FROM {schema}.{table}",
                        {
                            "execution_time_ms": execution_time_ms,
                            "rows_affected": row_count,
                            "execution_plan": {},
                        },
                        str(e),
                    )

                query = (
                    query[:start] + f"postgres_db.{schema}.{table}" + query[end + 1 :]
                )

            # Execute final query with EXPLAIN ANALYZE
            explain_result = self.duck_conn.execute(
                f"EXPLAIN ANALYZE {query}"
            ).fetchall()
            result = self.duck_conn.execute(query).fetchdf()

            # Log DuckDB performance
            self._log_query_performance(
                "duckdb",
                query,
                {
                    "execution_time_ms": (datetime.now() - start_time).total_seconds()
                    * 1000,
                    "rows_affected": len(result),
                    "execution_plan": {
                        "explain_analyze": [row[0] for row in explain_result]
                    },
                },
            )

            return result

        except Exception as e:
            # Log error if any
            self._log_query_performance(
                "duckdb",
                query,
                {
                    "execution_time_ms": (datetime.now() - start_time).total_seconds()
                    * 1000,
                    "rows_affected": 0,
                    "execution_plan": {},
                },
                str(e),
            )
            raise

    def _find_matching_parenthesis(self, text, start):
        """Find the matching closing parenthesis.

        Args:
            text: Text to search in
            start: Starting position of the opening parenthesis

        Returns:
            int: Position of the matching closing parenthesis

        Raises:
            ValueError: If no matching parenthesis is found
        """
        count = 1
        i = start + 1
        while i < len(text) and count > 0:
            if text[i] == "(":
                count += 1
            elif text[i] == ")":
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1
