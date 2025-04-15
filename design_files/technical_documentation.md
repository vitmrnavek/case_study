# Technical Documentation

## System Architecture

The system is built around three main components:
1. Database Integration Layer
2. Cross-Database Query Engine (DuckDB)
3. AI-Powered Join Advisor

### Component Diagram

```mermaid
graph TB
    A[Client Application] --> B[AI Join Advisor]
    A --> C[Database Integration Layer]
    A --> D[DuckDB Integration]
    
    B --> D
    C --> D
    
    C --> E[PostgreSQL]
    C --> F[BigQuery]
    
    D --> E
    D --> F
```

## Core Components Documentation

### 1. Database Integration Layer

#### ConnectionConfig Class
Manages database connection configurations and validation.

```mermaid
classDiagram
    class ConnectionConfig {
        +config_path: Path
        +connections: Dict[str, ConnectionInfo]
        +schema_path: Optional[str]
        +__init__(config_path: Union[str, Path])
        +_load_config()
        +get_connection(name: str)
        +list_connections()
        +validate()
        +to_dict()
        +save(path: Optional[Union[str, Path]])
    }
```

Key Methods:
- `_load_config()`: Loads and validates configuration from YAML file
- `get_connection()`: Retrieves connection information by name
- `validate()`: Validates the entire configuration
- `save()`: Saves configuration to YAML file

#### Database Class
Base class for database connections and operations.

```mermaid
classDiagram
    class Database {
        +connection_name: str
        +connection_params: Any
        +schema: Schema
        +__init__(connection_name: str)
        +connect(fetch_schema: bool)
        +gather_schema()
        +fetch_data(query: str)
    }
    
    class PostgresDatabase {
        +cursor: psycopg2.cursor
        +connection: psycopg2.connection
        +_connect_postgres()
        +fetch_data(query: str)
        +get_table_metadata(schema: str)
        +get_column_metadata(table: str, schema: str)
    }
    
    class BigQueryDatabase {
        +client: bigquery.Client
        +_connect_bigquery()
        +fetch_data(query: str)
        +get_schema_metadata()
    }
    
    Database <|-- PostgresDatabase
    Database <|-- BigQueryDatabase
```

### 2. DuckDB Integration Layer

#### DuckDBIntegrator Class
Handles cross-database operations using DuckDB.

```mermaid
classDiagram
    class DuckDBIntegrator {
        +duck_conn: duckdb.Connection
        +config: ConnectionConfig
        +__init__(config_path: str)
        +_setup_connections()
        +_setup_logging_table()
        +run_query(query: str)
        +get_query_logs(start_time: datetime, end_time: datetime)
    }
```

Key Methods:
- `_setup_connections()`: Initializes connections to PostgreSQL and BigQuery
- `run_query()`: Executes cross-database queries
- `get_query_logs()`: Retrieves query performance logs

### 3. AI Join Advisor

#### SmartJoinBuilder Class
Provides AI-powered join recommendations.

```mermaid
classDiagram
    class SmartJoinBuilder {
        +duck_integrator: DuckDBIntegrator
        +config: ConnectionConfig
        +databases: Dict[str, Database]
        +__init__(config_path: str)
        +initialize_connections(selected_connections: List[str])
        +suggest_join(table1: TableIdentifier, table2: TableIdentifier, use_case: str)
        +construct_duckdb_query(join_recommendation: Dict, table1: TableIdentifier, table2: TableIdentifier)
        +execute_join(query: str)
    }
```

## 4. Metadata System

The metadata system manages database schema information, caching, and schema updates. It's implemented in `metadata.py` and consists of several key classes that handle database metadata representation and management.

### Class Hierarchy

```mermaid
classDiagram
    class BaseModel{
        <<Pydantic>>
    }
    class Column{
        +str column_name
        +str column_comment
        +str data_type
        +bool is_nullable
        +str column_default
    }
    class Table{
        +str table_name
        +str schema_name
        +str table_comment
        +str ai_annotation
        +int rows_count
        +List[Column] columns
        +full_name()
    }
    class Schema{
        +str database_name
        +str database_type
        +List[Table] tables
        +datetime cache_timestamp
        +int cache_valid_days
        +from_json()
        +load()
        +save()
        +get_schema()
    }
    class Databases{
        +List[Schema] databases
        +str path
        +from_yaml()
        +to_yaml()
        +update_database()
    }
    
    BaseModel <|-- Column
    BaseModel <|-- Table
    BaseModel <|-- Schema
    BaseModel <|-- Databases
```

### Key Components

#### 1. Column Class
Represents a database column with its properties:
- `column_name`: Name of the column
- `column_comment`: Documentation/description of the column
- `data_type`: SQL data type
- `is_nullable`: Whether the column can contain NULL values
- `column_default`: Default value for the column

#### 2. Table Class
Represents a database table:
- `table_name`: Name of the table
- `schema_name`: Database schema name (defaults to "public" for PostgreSQL)
- `table_comment`: Documentation/description of the table
- `ai_annotation`: AI-generated annotations about the table
- `rows_count`: Number of rows in the table
- `columns`: List of Column objects



Key Methods:
- `from_json()`: Creates Schema object from JSON data
- `load()`: Loads schema from cache file with timestamp validation
- `save()`: Saves schema to cache file with updated timestamp
- `get_schema()`: Gets schema from cache or fetches fresh data

#### 4. Databases Class
Manages multiple database schemas:
- `databases`: List of Schema objects
- `path`: Default path for YAML storage
- `from_yaml()`: Creates Databases object from YAML file
- `to_yaml()`: Saves databases information to YAML file
- `update_database()`: Updates existing database schema or adds new one

### Caching System

```mermaid
graph TB
    A[Schema Request] --> B{Cache Exists?}
    B -->|Yes| C{Cache Valid?}
    B -->|No| D[Fetch Fresh Data]
    C -->|Yes| E[Return Cached Data]
    C -->|No| D
    D --> F[Save to Cache]
    F --> G[Return Fresh Data]
```

Features:
- Timestamp-based cache validation
- Configurable cache validity period
- Automatic cache refresh on expiration
- Force refresh option

### Schema Update Process

```mermaid
graph LR
    A[New Schema] --> B[Update Process]
    B --> C{Exists?}
    C -->|Yes| D[Merge Changes]
    C -->|No| E[Add New]
    D --> F[Preserve Values]
    E --> G[Save Changes]
    F --> G
```

The update process:
1. Recursively compares existing and new schemas
2. Preserves existing values if new values are empty
3. Updates only non-empty values
4. Handles nested structures and lists
5. Maintains data integrity during updates


## Data Flow Diagrams

### 1. Join Recommendation Flow

```mermaid
sequenceDiagram
    participant Client
    participant SmartJoinBuilder
    participant AI Advisor
    participant DuckDB
    participant Databases

    Client->>SmartJoinBuilder: Request Join Recommendation
    SmartJoinBuilder->>Databases: Fetch Schema Information
    SmartJoinBuilder->>Databases: Fetch Sample Data
    SmartJoinBuilder->>AI Advisor: Get Join Strategy
    AI Advisor-->>SmartJoinBuilder: Return Recommendation
    SmartJoinBuilder->>DuckDB: Construct Query
    DuckDB->>Databases: Execute Query
    Databases-->>DuckDB: Return Results
    DuckDB-->>SmartJoinBuilder: Process Results
    SmartJoinBuilder-->>Client: Return Final Results
```

### 2. Query Execution Flow

```mermaid
sequenceDiagram
    participant Client
    participant DuckDBIntegrator
    participant PostgreSQL
    participant BigQuery
    participant Logging

    Client->>DuckDBIntegrator: Execute Query
    DuckDBIntegrator->>DuckDBIntegrator: Parse Query
    
    alt PostgreSQL Data
        DuckDBIntegrator->>PostgreSQL: Fetch Data
        PostgreSQL-->>DuckDBIntegrator: Return Results
    else BigQuery Data
        DuckDBIntegrator->>BigQuery: Fetch Data
        BigQuery-->>DuckDBIntegrator: Return Results
    end

    DuckDBIntegrator->>DuckDBIntegrator: Process Results
    DuckDBIntegrator->>Logging: Log Performance
    DuckDBIntegrator-->>Client: Return Results
```

#### 3. Metadata Schema Caching
Manages database schema with caching capabilities:

```mermaid
sequenceDiagram
    participant Client
    participant Schema
    participant Cache
    participant Database
    
    Client->>Schema: get_schema(path, db_name)
    Schema->>Cache: load(path)
    
    alt Cache Valid
        Cache-->>Schema: Return cached schema
        Schema-->>Client: Return schema
    else Cache Invalid
        Schema->>Database: fetch_callback()
        Database-->>Schema: Fresh schema data
        Schema->>Cache: save(path)
        Schema-->>Client: Return fresh schema
    end
```


