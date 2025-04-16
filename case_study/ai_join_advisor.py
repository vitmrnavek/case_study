from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Optional, List
import json
from case_study.integrations.database import (
    ConnectionConfig,
    Database,
    PostgresDatabase,
    BigQueryDatabase,
    TableIdentifier,
)
from case_study.integrations.duckdb import DuckDBIntegrator
import pandas as pd
from datetime import datetime
from case_study.utils.logger import get_logger

import os
from case_study.utils.settings import DATABASE_CONFIG_PATH


# Initialize logger for join operations
logger = get_logger("join-advisor")


def determine_join_strategy(
    table1_description: str,
    table2_description: str,
    use_case_description: str,
    table1_sample_data: Optional[Dict] = None,
    table2_sample_data: Optional[Dict] = None,
) -> Dict:
    """Determine the optimal join strategy between two tables using LangChain and GPT-4.

    Args:
        table1_description: Description of the first table including schema and metadata
        table2_description: Description of the second table including schema and metadata
        use_case_description: Description of the intended use case for joining the tables
        table1_sample_data: Optional sample data from the first table
        table2_sample_data: Optional sample data from the second table

    Returns:
        Dict: JSON structured output containing join strategy details including:
            - join_strategy: Type of join to use
            - left_table: Name of the left table
            - right_table: Name of the right table
            - left_join_columns: Columns to join on from left table
            - right_join_columns: Columns to join on from right table
            - explanation: Detailed explanation of the recommendation
            - additional_considerations: Data quality and performance considerations
    """

    prompt_template = """You are a data expert helping to determine the optimal join strategy between two tables.

    Table 1 Description:
    {table1_description}

    Table 2 Description:
    {table2_description}

    Use Case:
    {use_case_description}

    Sample Data from Table 1 (if available):
    {table1_sample}

    Sample Data from Table 2 (if available):
    {table2_sample}

    Consider the following join strategies:
    1. FULL OUTER JOIN - Use when you need all records from both tables, good for data reconciliation
    2. LEFT JOIN - Use when Table 1 is the main table and Table 2 provides supplementary data
    3. INNER JOIN - Use when you only want matching records and the relationship is mandatory
    4. UNION - Use when tables have the same structure but different records (no joining key needed)

    Based on the provided information, analyze the tables and provide a recommendation in the following JSON format:
    {{
        "join_strategy": "LEFT JOIN|INNER JOIN|FULL OUTER JOIN|UNION",
        "left_table": "name of the left table",
        "left_join_columns": ["column1", "column2"],
        "right_table": "name of the right table",
        "right_join_columns": ["column1", "column2"],
        "explanation": "Detailed explanation of why this join strategy is recommended",
        "additional_considerations": {{
            "data_quality_checks": ["check1", "check2"],
            "performance_considerations": ["consideration1", "consideration2"],
            "alternative_strategies": ["alternative1", "alternative2"]
        }}
    }}

    Ensure the response is valid JSON format and includes all fields. The join columns should be actual columns from the table descriptions.
    The explanation should be detailed but concise.
    """

    # Create the prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Initialize the LLM with API key from environment
    llm = ChatOpenAI(
        temperature=0,  # Use 0 for more deterministic responses
        model_name="gpt-4",  # Using GPT-4 for better reasoning
        openai_api_key=os.getenv("OPENAI_API_KEY"),  # Get API key from environment
        base_url="https://api.openai.com/v1",  # Set base URL directly
    )

    # Create the chain using the new RunnableSequence approach
    chain = prompt | llm

    # Run the chain with the new invoke method
    response = chain.invoke(
        {
            "table1_description": table1_description,
            "table2_description": table2_description,
            "use_case_description": use_case_description,
            "table1_sample": (
                table1_sample_data if table1_sample_data else "No sample data provided"
            ),
            "table2_sample": (
                table2_sample_data if table2_sample_data else "No sample data provided"
            ),
        }
    )

    # Parse the response to ensure it's valid JSON
    try:
        # Extract the content from the response
        content = response.content if hasattr(response, "content") else str(response)
        json_response = json.loads(content)
        return json_response
    except json.JSONDecodeError:
        # Fallback response if JSON parsing fails
        return {
            "error": "Failed to generate valid JSON response",
            "raw_response": str(response),
        }


class SmartJoinBuilder:
    def __init__(self, config_path=DATABASE_CONFIG_PATH):
        """Initialize SmartJoinBuilder with database connections and DuckDB integrator.

        Args:
            config_path: Path to the database configuration file
        """
        logger.info(f"Initializing SmartJoinBuilder with config path: {config_path}")
        # Initialize DuckDB integrator
        self.duck_integrator = DuckDBIntegrator(config_path)

        # Load connections from ConnectionConfig
        self.config = ConnectionConfig(config_path)

        self.databases: Dict[str, Database] = {}
        self.available_tables: List[TableIdentifier] = []
        logger.info("SmartJoinBuilder initialized successfully")

    def initialize_connections(self, selected_connections: List[str]):
        """Initialize database connections and fetch their schemas.

        Args:
            selected_connections: List of connection names to initialize
        """
        logger.info(f"Initializing connections for: {selected_connections}")
        initialized_count = 0

        for conn_name in selected_connections:
            try:
                logger.info(f"Setting up connection '{conn_name}'")
                conn_info = self.config.get_connection(conn_name)

                # Create appropriate database instance
                if conn_info.type == "postgres":
                    logger.info(
                        f"Creating PostgreSQL database instance for {conn_name}"
                    )
                    db = PostgresDatabase(conn_name)
                elif conn_info.type == "bigquery":
                    logger.info(f"Creating BigQuery database instance for {conn_name}")
                    db = BigQueryDatabase(conn_name)
                else:
                    logger.warning(
                        f"Unsupported database type for connection {conn_name}"
                    )
                    continue

                # Connect and fetch schema
                logger.info(
                    f"Connecting to database and fetching schema for {conn_name}"
                )
                db.connect()
                self.databases[conn_name] = db

                # Add tables to available_tables list
                table_count = 0
                if hasattr(db, "schema") and hasattr(db.schema, "tables"):
                    for table in db.schema.tables:
                        table_id = TableIdentifier(
                            connection_name=conn_name,
                            schema_name=table.schema_name,
                            database_type=conn_info.type,
                            database_name=db.schema.database_name,
                            table_name=table.table_name,
                        )
                        self.available_tables.append(table_id)
                        table_count += 1

                logger.info(f"Successfully added {table_count} tables from {conn_name}")
                initialized_count += 1

            except Exception as e:
                logger.error(f"Failed to initialize connection {conn_name}: {str(e)}")

        logger.info(
            f"Connection initialization completed. Successfully initialized {initialized_count}/{len(selected_connections)} connections"
        )

    def _format_table_description(self, table: TableIdentifier) -> str:
        """Format table metadata into a description for the AI advisor.

        Args:
            table: TableIdentifier object containing table information

        Returns:
            str: Formatted description of the table including columns and metadata

        Raises:
            ValueError: If the table is not found in the schema
        """
        logger.info(f"Formatting table description for {table.full_name}")
        db = self.databases[table.connection_name]
        table_schema = None

        # Find the table in the database schema
        for db_table in db.schema.tables:
            if db_table.table_name == table.table_name:
                table_schema = db_table
                break

        if not table_schema:
            logger.error(f"Table {table.full_name} not found in schema")
            raise ValueError(f"Table {table.full_name} not found in schema")

        description = f"""
        {table.full_name}:
        - Primary key: Unknown  # Primary key information not available in schema
        - Columns:
        """

        column_count = 0
        for column in table_schema.columns:
            description += f"""    * {column.column_name} ({column.data_type})
                               {f"- {column.column_comment}" if column.column_comment else ''}\n"""
            column_count += 1

        if table_schema.table_comment:
            description += f"\nTable description: {table_schema.table_comment}"

        logger.info(
            f"Generated description for {table.full_name} with {column_count} columns"
        )
        return description

    def suggest_join(
        self, table1: TableIdentifier, table2: TableIdentifier, use_case: str
    ) -> Dict:
        """Get join recommendation from AI advisor with sample data.

        Args:
            table1: TableIdentifier for the first table
            table2: TableIdentifier for the second table
            use_case: Description of the intended use case for joining the tables

        Returns:
            Dict: Join recommendation including strategy and implementation details
        """
        logger.info(
            f"Generating join suggestion for tables: {table1.full_name} and {table2.full_name}"
        )
        logger.info(f"Use case: {use_case}")

        table1_desc = self._format_table_description(table1)
        table2_desc = self._format_table_description(table2)
        tables_dict = {1: table1, 2: table2}
        table_refs = {}

        # Fetch sample data from both tables
        try:
            # Construct sample queries (limit to 5 rows)
            # Format table references properly for DuckDB
            for i in range(1, 3):
                if tables_dict[i].database_type == "postgres":
                    table_refs[i] = (
                        f"POSTGRES({tables_dict[i].schema_name}.{tables_dict[i].table_name})"
                    )
                elif tables_dict[i].database_type == "bigquery":
                    table_refs[i] = (
                        f"BIGQUERY({tables_dict[i].database_name}.{tables_dict[i].schema_name}.{tables_dict[i].table_name})"
                    )

            sample_query1 = f"SELECT * FROM {table_refs[1]} LIMIT 5"
            sample_query2 = f"SELECT * FROM {table_refs[2]} LIMIT 5"

            logger.info(
                f"Fetching sample data with queries:\n{sample_query1}\n{sample_query2}"
            )

            # Fetch sample data
            table1_sample = self.duck_integrator.run_query(sample_query1)
            table2_sample = self.duck_integrator.run_query(sample_query2)

            logger.info(
                f"Successfully fetched sample data: {len(table1_sample)} rows from table1, {len(table2_sample)} rows from table2"
            )

            # Convert sample data to dictionary format
            table1_sample_dict = {
                "columns": list(table1_sample.columns),
                "sample_rows": table1_sample.to_dict(orient="records"),
            }
            table2_sample_dict = {
                "columns": list(table2_sample.columns),
                "sample_rows": table2_sample.to_dict(orient="records"),
            }

        except Exception as e:
            logger.warning(f"Failed to fetch sample data: {str(e)}")
            table1_sample_dict = {"error": "Failed to fetch sample data"}
            table2_sample_dict = {"error": "Failed to fetch sample data"}

        logger.info("Calling AI advisor for join strategy recommendation")
        recommendation = determine_join_strategy(
            table1_description=table1_desc,
            table2_description=table2_desc,
            use_case_description=use_case,
            table1_sample_data=table1_sample_dict,
            table2_sample_data=table2_sample_dict,
        )

        logger.info(
            f"Received join recommendation: {recommendation.get('join_strategy', 'Unknown')} join"
        )
        return recommendation

    def construct_duckdb_query(
        self,
        join_recommendation: Dict,
        table1: TableIdentifier,
        table2: TableIdentifier,
    ) -> str:
        """Construct a DuckDB query based on the join recommendation.

        Args:
            join_recommendation: Dictionary containing join strategy and details
            table1: TableIdentifier for the first table
            table2: TableIdentifier for the second table

        Returns:
            str: DuckDB SQL query implementing the recommended join
        """
        logger.info("Constructing DuckDB query from join recommendation")
        join_type = join_recommendation["join_strategy"]
        logger.info(f"Join type: {join_type}")

        if join_type == "UNION":
            logger.info("Constructing UNION query")
            query = f"""
            WITH 
                t1 AS (SELECT * FROM {table1.get_duckdb_reference()}),
                t2 AS (SELECT * FROM {table2.get_duckdb_reference()})
            SELECT * FROM t1
            UNION
            SELECT * FROM t2
            """
        else:
            # Get the join conditions
            join_conditions = " AND ".join(
                [
                    f"t1.{left} = t2.{right}"
                    for left, right in zip(
                        join_recommendation["left_join_columns"],
                        join_recommendation["right_join_columns"],
                    )
                ]
            )
            logger.info(f"Join conditions: {join_conditions}")

            query = f"""
            WITH 
                t1 AS (SELECT * FROM {table1.get_duckdb_reference()}),
                t2 AS (SELECT * FROM {table2.get_duckdb_reference()})
            SELECT 
                t1.*,
                t2.*
            FROM t1
            {join_type} t2
                ON {join_conditions}
            """

        logger.info("Query construction completed")
        return query

    def execute_join(self, query: str) -> pd.DataFrame:
        """Execute a join query using DuckDB.

        Args:
            query: SQL query to execute

        Returns:
            pd.DataFrame: Result of the join operation
        """
        logger.info("Executing join query")
        try:
            result = self.duck_integrator.execute_analysis(query)
            logger.info(
                f"Join query executed successfully. Result shape: {result.shape}"
            )
            return result
        except Exception as e:
            logger.error(f"Error executing join query: {str(e)}")
            raise

    def get_performance_logs(
        self, start_time: datetime = None, end_time: datetime = None
    ) -> pd.DataFrame:
        """Retrieve query performance logs for a specific time range.

        Args:
            start_time: Optional start time to filter logs
            end_time: Optional end time to filter logs

        Returns:
            pd.DataFrame: Performance logs including query details and execution metrics
        """
        logger.info(
            f"Retrieving performance logs (start_time={start_time}, end_time={end_time})"
        )
        return self.duck_integrator.get_query_logs(
            start_time=start_time, end_time=end_time
        )


# Example usage:
if __name__ == "__main__":
    # Example case
    table1_desc = """
    Customer Orders table (orders):
    - Primary key: order_id
    - Columns: 
        * order_id (integer)
        * customer_id (integer)
        * order_date (timestamp)
        * total_amount (decimal)
    - Every order must have a customer
    """

    table2_desc = """
    Customer Reviews table (reviews):
    - Primary key: review_id
    - Columns:
        * review_id (integer)
        * order_id (integer)
        * rating (integer)
        * review_text (text)
    - Not all orders have reviews
    """

    use_case = """
    We need to analyze all orders and include review information where available.
    We don't want to lose any order information, but it's okay if some orders
    don't have reviews. We'll be using this for a customer satisfaction analysis.
    """

    result = determine_join_strategy(
        table1_description=table1_desc,
        table2_description=table2_desc,
        use_case_description=use_case,
    )

    # Pretty print the JSON result
    print(json.dumps(result, indent=2))
    # Get performance logs
