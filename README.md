# Cross Database Query Engine

This project provides a powerful data integration and analysis toolkit that enables seamless interaction with multiple database systems (PostgreSQL and BigQuery) and includes smart join recommendations using AI.

## Features

- Multi-database support (PostgreSQL and BigQuery)
- Smart join recommendations using AI/LLM
- Schema management and validation
- Query performance logging
- DuckDB integration for efficient cross-database operations
- Comprehensive logging system

## Prerequisites

- Python 3.8+
- PostgreSQL (if using PostgreSQL connections)
- Google Cloud account and credentials (if using BigQuery)
- OpenAI API key (for smart join recommendations)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up configuration:
   - Copy `config.example.yaml` to `config.yaml`
   - Update database connection parameters
   - Set up environment variables:
     ```bash
     export OPENAI_API_KEY=your_api_key
     ```

## Configuration

### Database Configuration

Create a configuration file at `config.yaml` with your database connections (see and copy `config_example.yaml` for your first use case  ):

```yaml
databases:
  postgres_connection:
    type: postgres
    connection_params:
      host: localhost
      port: 5432
      database: your_database
      user: your_user
      password: your_password

  bigquery_connection:
    type: bigquery
    connection_params:
      project_id: your-project-id
      location: your-location
      credentials_path: path/to/credentials.json
```

## Usage

### Basic Database Operations

```python
from case_study.integrations.database import Database, ConnectionConfig

# Initialize database connection
db = Database("postgres_connection")
db.connect()

# Execute queries
query = "SELECT * FROM your_table"
result = db.fetch_data(query)
```

### Smart Join Operations

```python
from case_study.ai_join_advisor import SmartJoinBuilder

# Initialize the join builder
join_builder = SmartJoinBuilder()
join_builder.initialize_connections(["postgres_connection", "bigquery_connection"])

# Get join recommendation
table1 = TableIdentifier(
    connection_name="postgres_connection",
    database_type="postgres",
    database_name="your_db",
    schema_name="public",
    table_name="table1"
)

table2 = TableIdentifier(
    connection_name="bigquery_connection",
    database_type="bigquery",
    database_name="your_project",
    schema_name="your_dataset",
    table_name="table2"
)

use_case = "Describe your use case here"
recommendation = join_builder.suggest_join(table1, table2, use_case)

# Execute the recommended join
query = join_builder.construct_duckdb_query(recommendation, table1, table2)
result = join_builder.execute_join(query)
```
Or you can just use main.ipynb jupyter notebook!

### Cross-Database Operations with DuckDB

```python
from case_study.integrations.duckdb import DuckDBIntegrator

# Initialize DuckDB integrator
duck = DuckDBIntegrator()

# Execute cross-database query
query = """
WITH 
    pg_data AS (
        SELECT * FROM POSTGRES('schema.table1')
    ),
    bq_data AS (
        SELECT * FROM BIGQUERY('project.dataset.table2')
    )
SELECT * FROM pg_data JOIN bq_data ON pg_data.id = bq_data.id
"""
result = duck.run_query(query)
```

## Logging

The project includes comprehensive logging for all operations:

- Database operations are logged in the database logger
- DuckDB operations are logged in the DuckDB logger
- Join operations are logged in the join-advisor logger

Logs include:
- Query execution times
- Row counts
- Error messages
- Performance statistics

## Performance Monitoring

You can monitor query performance using the built-in logging system:

```python
# Get performance logs
from datetime import datetime, timedelta

start_time = datetime.now() - timedelta(hours=1)
logs = duck.get_query_logs(start_time=start_time)
```

## Error Handling

The system includes comprehensive error handling and logging. All operations are logged with appropriate error messages and stack traces when issues occur.

## Best Practices

1. Always validate database configurations before running operations
2. Use the smart join advisor for complex joins between different databases
3. Monitor query performance using the built-in logging system
4. Keep credentials secure and use environment variables for sensitive information
5. Regularly check logs for potential issues or performance bottlenecks

## Troubleshooting

Common issues and solutions:

1. Connection errors:
   - Verify database credentials
   - Check network connectivity
   - Ensure proper permissions

2. Performance issues:
   - Check query logs for slow operations
   - Verify proper indexes are in place
   - Consider optimizing join conditions

3. BigQuery authentication:
   - Verify credentials file path
   - Check project permissions
   - Ensure proper service account setup
