import yaml
from pydantic import BaseModel, Field
from typing import List, Dict
import logging

class Column(BaseModel):
    column_name: str
    column_comment: str = Field(default="")
    data_type: str
    is_nullable: bool = Field(default=False)
    column_default: str

class Table(BaseModel):
    table_name: str
    table_comment: str = Field(default="")
    ai_annotation: str = Field(default="")
    rows_count: int 
    columns: List[Column]

class Schema(BaseModel):
    database_name: str
    database_type: str
    tables: List[Table]

    @classmethod
    def from_json(cls, json_data: Dict) -> 'Schema':
        # Create Schema object
        return cls(
            database_name=json_data['database_name'],
            database_type=json_data['database_type'],
            tables=[
                Table(
                    table_name=table_info['table_name'],
                    table_comment="" if table_info['table_comment'] is None else table_info['table_comment'],
                    rows_count=0 if table_info.get('rows_count', 0) is None else table_info.get('rows_count', 0),
                    columns=[
                        Column(
                            column_name=column['column_name'],
                            column_comment="" if column['column_comment'] is None else column['column_comment'],
                            data_type=column['data_type'],
                            is_nullable=column.get('is_nullable', False),
                            column_default="" if column['column_default'] is None else column['column_default']
                        )
                        for column in table_info['columns']
                    ]
                )
                for table_info in json_data['tables']
            ]
        )
    
    @classmethod
    def load(cls,path: str) :
        with open(path, 'r') as f:
            file_output = yaml.safe_load(f)
            file_output["databases"] = [cls.from_json(i) for i in file_output["databases"]]
        return file_output
    
    def save(self,path:str):
        
            try:
                with open(path, 'r') as read_file:
                    data = yaml.load(read_file,yaml.SafeLoader)

                    if self.database_name not in [i.get('database_name') for i in data['databases']]:
                        data['databases'].append(self.model_dump())
                    else:
                        data['databases'][self.database_name] = update_dict_recursively(data['databases'][self.database_name], self.model_dump())
                with open(path, 'w') as write_file:
                    yaml.dump(data, write_file, default_flow_style=False,)
            except Exception as e:
                logging.warning(f"Database {self.database_name} not found in {path}, creating new entry")
                with open(path, 'w') as write_file:
                    yaml.dump({'databases':[self.model_dump()]}, write_file, default_flow_style=False)


        

class Databases(BaseModel):
    databases: List[Schema]
    path: str = Field(default="databases.yaml")

    @classmethod
    def from_yaml(cls, file_path: str) -> 'Databases':
        with open(file_path, 'r') as f:
            file_output = yaml.safe_load(f)
        return cls.from_json(file_output)


    def to_yaml(self, file_path: str = None) -> None:
        """
        Save the databases information to a YAML file.
        
        Args:
            file_path: Optional path to save the YAML file. If not provided, uses the default path.
        """
        if file_path is None:
            file_path = self.path
            
        with open(file_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    def update_database(self, schema: Schema) -> None:
        """
        Update an existing database schema or add a new one if it doesn't exist.
        Preserves existing values if the new schema has empty or missing values.
        
        Args:
            schema: The Schema object to update or add
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
    
def update_dict_recursively(old_dict: dict, new_dict: dict) -> dict:
    """
    Recursively update a dictionary with values from a new dictionary.
    Only overwrites values that are not empty or None in the new dictionary.
    
    Args:
        old_dict: The original dictionary to update
        new_dict: The new dictionary with values to update
        
    Returns:
        The updated dictionary
    """
    result = old_dict.copy()
    
    for key, new_value in new_dict.items():
        # If key doesn't exist in old dict, add it
        if key not in result:
            result[key] = new_value
            continue
            
        # If the value is a dictionary, recursively update
        if isinstance(new_value, dict) and isinstance(result[key], dict):
            result[key] = update_dict_recursively(result[key], new_value)
        # If the value is a list of dictionaries, update each item
        elif isinstance(new_value, list) and isinstance(result[key], list):
            # For lists, we need a way to match items - using first field as identifier if possible
            if all(isinstance(item, dict) for item in new_value + result[key]):
                # Create a map of existing items by their first key (assuming it's an identifier)
                if len(result[key]) > 0 and len(new_value) > 0:
                    id_key = next(iter(result[key][0].keys()))
                    existing_map = {item.get(id_key): item for item in result[key] if id_key in item}
                    
                    # Update existing items and add new ones
                    updated_list = []
                    for new_item in new_value:
                        if id_key in new_item and new_item[id_key] in existing_map:
                            # Update existing item
                            updated_item = update_dict_recursively(
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


