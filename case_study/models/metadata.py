import yaml
from pydantic import BaseModel, Field
from typing import List, Dict, Callable
from case_study.utils.logger import get_logger
import os
from datetime import datetime


# Initialize logger for database operations
logger = get_logger("metadata")


class Column(BaseModel):
    """Model representing a database column.

    Attributes:
        column_name: Name of the column
        column_comment: Optional comment describing the column
        data_type: SQL data type of the column
        is_nullable: Whether the column can contain NULL values
        column_default: Default value for the column
    """

    column_name: str
    column_comment: str = Field(default="")
    data_type: str
    is_nullable: bool = Field(default=False)
    column_default: str


class Table(BaseModel):
    """Model representing a database table.

    Attributes:
        table_name: Name of the table
        schema_name: Name of the schema containing the table
        table_comment: Optional comment describing the table
        ai_annotation: Optional AI-generated annotation about the table
        rows_count: Number of rows in the table
        columns: List of columns in the table
    """

    table_name: str
    schema_name: str = Field(default="public")  # Default schema for PostgreSQL
    table_comment: str = Field(default="")
    ai_annotation: str = Field(default="")
    rows_count: int
    columns: List[Column]

    @property
    def full_name(self) -> str:
        """Get the full table name including schema.

        Returns:
            str: Full table name in format 'schema.table'
        """
        return f"{self.schema_name}.{self.table_name}"


class Schema(BaseModel):
    """Model representing a database schema.

    Attributes:
        database_name: Name of the database
        database_type: Type of database (e.g., 'postgres', 'bigquery')
        tables: List of tables in the schema
        cache_timestamp: When the schema was last cached
        cache_valid_days: How long the cache remains valid
    """

    database_name: str
    database_type: str
    tables: List[Table]
    cache_timestamp: datetime = Field(default_factory=datetime.now)
    cache_valid_days: int = 1  # Default cache validity period in days

    @classmethod
    def from_json(cls, json_data: Dict) -> "Schema":
        """Create a Schema instance from JSON data.

        Args:
            json_data: Dictionary containing schema information

        Returns:
            Schema: New Schema instance with parsed data
        """
        # Create Schema object with timestamp if available
        return cls(
            database_name=json_data["database_name"],
            database_type=json_data["database_type"],
            cache_timestamp=datetime.fromisoformat(
                json_data.get("cache_timestamp", datetime.now().isoformat())
            ),
            tables=[
                Table(
                    table_name=table_info["table_name"],
                    schema_name=table_info.get("schema_name", "public"),
                    table_comment=(
                        ""
                        if table_info["table_comment"] is None
                        else table_info["table_comment"]
                    ),
                    rows_count=(
                        0
                        if table_info.get("rows_count", 0) is None
                        else table_info.get("rows_count", 0)
                    ),
                    columns=[
                        Column(
                            column_name=column["column_name"],
                            column_comment=(
                                ""
                                if column["column_comment"] is None
                                else column["column_comment"]
                            ),
                            data_type=column["data_type"],
                            is_nullable=column.get("is_nullable", False),
                            column_default=(
                                ""
                                if column["column_default"] is None
                                else column["column_default"]
                            ),
                        )
                        for column in table_info["columns"]
                    ],
                )
                for table_info in json_data["tables"]
            ],
        )

    @classmethod
    def load(cls, path: str, force_refresh: bool = False) -> Dict:
        """Load schema from cache file with timestamp validation.

        Args:
            path: Path to the cache file
            force_refresh: If True, ignore cache validity and force a refresh

        Returns:
            Dict: Dictionary containing database schemas
        """
        if not os.path.exists(path):
            return {"databases": []}

        with open(path, "r") as f:
            cache_data = yaml.safe_load(f)

        if not cache_data:
            return {"databases": []}

        # Convert cached data to Schema objects with timestamp validation
        valid_schemas = []
        for schema_data in cache_data.get("databases", []):
            schema = cls.from_json(schema_data)
            cache_age = datetime.now() - schema.cache_timestamp

            # Check if cache is still valid
            if not force_refresh and cache_age.days < schema.cache_valid_days:
                valid_schemas.append(schema)
            else:
                logger.info(
                    f"Cache expired for database {schema.database_name}, will be refreshed"
                )

        cache_data["databases"] = valid_schemas
        return cache_data

    def save(self, path: str):
        """Save schema to cache file with updated timestamp.

        Args:
            path: Path to the cache file
        """
        self.cache_timestamp = datetime.now()
        schema_data = self.model_dump()
        schema_data["cache_timestamp"] = self.cache_timestamp.isoformat()
        logger.info(f"Saving schema for database '{self.database_name}'")

        try:
            if os.path.exists(path):
                with open(path, "r") as read_file:
                    data = yaml.load(read_file, yaml.SafeLoader) or {"databases": []}

                    # Update or append schema with new timestamp
                    db_names = [db.get("database_name") for db in data["databases"]]
                    if self.database_name in db_names:
                        idx = db_names.index(self.database_name)
                        data["databases"][idx] = schema_data
                    else:
                        data["databases"].append(schema_data)
            else:
                data = {"databases": [schema_data]}

            with open(path, "w") as write_file:
                yaml.dump(data, write_file, default_flow_style=False)
                logger.info(f"Schema saved with timestamp {self.cache_timestamp}")

        except Exception as e:
            logger.warning(
                f"Error saving cache for database {self.database_name}: {str(e)}"
            )
            with open(path, "w") as write_file:
                yaml.dump(
                    {"databases": [schema_data]}, write_file, default_flow_style=False
                )

    @classmethod
    def get_schema(
        cls,
        path: str,
        database_name: str,
        fetch_callback: Callable[[], "Schema"],
        force_refresh: bool = False,
    ) -> "Schema":
        """Get schema from cache or fetch fresh data if cache is invalid.

        Args:
            path: Path to the cache file
            database_name: Name of the database to get schema for
            fetch_callback: Callback function to fetch fresh schema data
            force_refresh: If True, ignore cache and force fresh fetch

        Returns:
            Schema: Schema object for the requested database
        """
        cache_data = cls.load(path, force_refresh)
        # Set up logging for cache operations

        logger.info(
            f"Retrieving schema for database '{database_name}' (force_refresh={force_refresh})"
        )

        # Try to find schema in cache
        for schema in cache_data.get("databases", []):
            if schema.database_name == database_name:
                cache_age = (datetime.now() - schema.cache_timestamp).days
                if not force_refresh and cache_age < schema.cache_valid_days:
                    logger.info(
                        f"Cache hit for '{database_name}' (cache_age={cache_age} days)"
                    )
                    return schema

        # Cache miss or invalid - fetch fresh data
        fresh_schema = fetch_callback()
        logger.info(f"Fetched fresh schema for '{database_name}'")
        fresh_schema.save(path)
        return fresh_schema


class Databases(BaseModel):
    """Model representing a collection of database schemas.

    Attributes:
        databases: List of Schema objects
        path: Path to the YAML file storing the schemas
    """

    databases: List[Schema]
    path: str = Field(default="databases.yaml")

    @classmethod
    def from_yaml(cls, file_path: str) -> "Databases":
        """Create a Databases instance from a YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            Databases: New Databases instance with loaded data
        """
        with open(file_path, "r") as f:
            file_output = yaml.safe_load(f)
        return cls.from_json(file_output)

    def to_yaml(self, file_path: str = None) -> None:
        """Save the databases information to a YAML file.

        Args:
            file_path: Optional path to save the YAML file. If not provided, uses the default path.
        """
        if file_path is None:
            file_path = self.path

        with open(file_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def update_database(self, schema: Schema) -> None:
        """Update an existing database schema or add a new one.

        Args:
            schema: The Schema object to update or add

        Note:
            Preserves existing values if the new schema has empty or missing values.
        """
        # Check if database already exists
        for i, db in enumerate(self.databases):
            if db.database_name == schema.database_name:
                # Get dictionaries for both schemas
                existing_dict = db.model_dump()
                new_dict = schema.model_dump()

                # Recursively update the dictionary
                updated_dict = self._update_dict_recursively(existing_dict, new_dict)

                # Create updated schema from the merged dictionary
                self.databases[i] = Schema.model_validate(updated_dict)
                return

        # If we get here, the database doesn't exist, so add it
        self.databases.append(schema)

    def _update_dict_recursively(self, old_dict: dict, new_dict: dict) -> dict:
        """Recursively update a dictionary while preserving existing values.

        Args:
            old_dict: Original dictionary
            new_dict: Dictionary with new values

        Returns:
            dict: Updated dictionary with preserved values
        """
        result = old_dict.copy()

        for key, new_value in new_dict.items():
            # If key doesn't exist in old dict, add it
            if key not in result:
                result[key] = new_value
                continue

            # If the value is a dictionary, recursively update
            if isinstance(new_value, dict) and isinstance(result[key], dict):
                result[key] = self._update_dict_recursively(result[key], new_value)
            # If the value is a list of dictionaries, update each item
            elif isinstance(new_value, list) and isinstance(result[key], list):
                # For lists, we need a way to match items - using first field as identifier if possible
                if all(isinstance(item, dict) for item in new_value + result[key]):
                    # Create a map of existing items by their first key (assuming it's an identifier)
                    if len(result[key]) > 0 and len(new_value) > 0:
                        id_key = next(iter(result[key][0].keys()))
                        existing_map = {
                            item.get(id_key): item
                            for item in result[key]
                            if id_key in item
                        }

                        # Update existing items and add new ones
                        updated_list = []
                        for new_item in new_value:
                            if id_key in new_item and new_item[id_key] in existing_map:
                                # Update existing item
                                updated_item = self._update_dict_recursively(
                                    existing_map[new_item[id_key]], new_item
                                )
                                updated_list.append(updated_item)
                                del existing_map[new_item[id_key]]
                            else:
                                # Add new item
                                updated_list.append(new_item)

                        # Add remaining existing items
                        updated_list.extend(existing_map.values())
                        result[key] = updated_list
                    else:
                        # If either list is empty, use the new list
                        result[key] = new_value
                else:
                    # For non-dict lists, replace if new value is not empty
                    if new_value:
                        result[key] = new_value
            # For other types, only update if the new value is not empty/None
            elif new_value is not None and new_value != "":
                result[key] = new_value

        return result
