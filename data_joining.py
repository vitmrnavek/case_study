# data_joining.py

import pandas as pd


def validate_inputs(join_type,handle_duplicates,join_on):
    valid_join_types = {'inner', 'outer', 'left', 'right'}
    if join_type not in valid_join_types:
        raise ValueError(f"Invalid join_type: {join_type}. Must be one of {valid_join_types}.")

    # Validate handle_duplicates
    valid_handle_duplicates = {'merge', 'keep_first', 'keep_last', 'keep_both'}
    if handle_duplicates not in valid_handle_duplicates:
        raise ValueError(f"Invalid handle_duplicates: {handle_duplicates}. Must be one of {valid_handle_duplicates}.")

    # Validate join_on
    if isinstance(join_on, str):
        left_on = right_on = join_on
    elif isinstance(join_on, list) and all(isinstance(pair, tuple) and len(pair) == 2 for pair in join_on):
        left_on, right_on = zip(*join_on)
    else:
        raise ValueError("join_on must be a string or a list of tuples, each containing two elements.")
    
    return left_on, right_on

def validate_join_on(df1,df2,left_on,right_on):
    if isinstance(left_on, str):
        if left_on not in df1.columns or right_on not in df2.columns:
            raise ValueError(f"Column '{left_on}' must be present in both dataframes.")
        # Check if data types match
        if df1[left_on].dtype != df2[right_on].dtype:
            raise ValueError(f"Data type mismatch for column '{left_on}': {df1[left_on].dtype} vs {df2[right_on].dtype}")
    else:
        missing_columns = [col for col in left_on if col not in df1.columns] + [col for col in right_on if col not in df2.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in dataframes: {missing_columns}")
        # Check if data types match
        for l_col, r_col in zip(left_on, right_on):
            if df1[l_col].dtype != df2[r_col].dtype:
                raise ValueError(f"Data type mismatch for columns '{l_col}' and '{r_col}': {df1[l_col].dtype} vs {df2[r_col].dtype}")

def calculate_match_stats(df1, df2, left_on, right_on):
    initial_rows_df1 = len(df1)
    initial_rows_df2 = len(df2)
    # Calculate the number of matched rows by counting join keys present in both tables
    if isinstance(left_on, str):
        # For single column join
        matched_rows = len(df1[df1[left_on].isin(df2[right_on])])
    else:
        # For multi-column join, create a temporary key in both dataframes
        df1_temp = df1.copy()
        df2_temp = df2.copy()
        df1_temp['_temp_key'] = df1_temp[list(left_on)].apply(tuple, axis=1)
        df2_temp['_temp_key'] = df2_temp[list(right_on)].apply(tuple, axis=1)
        matched_rows = len(df1_temp[df1_temp['_temp_key'].isin(df2_temp['_temp_key'])])
        
    matched_percentage_df1 = (matched_rows / initial_rows_df1) * 100 if initial_rows_df1 > 0 else 0
    matched_percentage_df2 = (matched_rows / initial_rows_df2) * 100 if initial_rows_df2 > 0 else 0
    # Log or print statistics
    stats = {
        'initial_rows_df1': initial_rows_df1,
        'initial_rows_df2': initial_rows_df2,
        'matched_rows': matched_rows,
        'matched_percentage_df1': matched_percentage_df1,
        'matched_percentage_df2': matched_percentage_df2
    }

    return stats





def join_data(df1, df2, join_on, join_type='inner', handle_duplicates='keep_first'):
    ### Validate join_type
    left_on, right_on = validate_inputs(join_type,handle_duplicates,join_on)
    ### Validate join_on columns are present in both dataframes and have the same data type
    validate_join_on(df1,df2,left_on,right_on)
    
    
    
    # Perform the join
    joined_df = df1.merge(df2, left_on=left_on, right_on=right_on, how=join_type)
    
    duplicate_columns = df1.columns.intersection(df2.columns)
    if handle_duplicates == 'merge':
        # Implement logic to merge duplicate columns
        pass
    elif handle_duplicates == 'keep_first':
        # Drop duplicate columns from the second dataframe
        pass
    elif handle_duplicates == 'keep_last':
        # Drop duplicate columns from the first dataframe
        pass
    elif handle_duplicates == 'keep_both':
        # Keep both duplicate columns
        pass
    match_stats = calculate_match_stats(df1,df2,left_on,right_on)
    print(match_stats)
    # Return joined dataframe and statistics
    return joined_df