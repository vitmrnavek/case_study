import ipywidgets as widgets
from IPython.display import display


def select_databases_widget(connections, builder):
    """Create and display a widget for selecting database connections.

    Args:
        connections: ConnectionConfig object containing available database connections
        builder: SmartJoinBuilder instance to initialize selected connections

    Returns:
        select_widget: A widget for selecting database connections
    """
    # Create a multiple selection widget for databases
    connection_names = connections.list_connections()
    options = [
        (f"{conn_name} ({connections.get_connection(conn_name).type})", conn_name)
        for conn_name in connection_names
    ]

    select_widget = widgets.SelectMultiple(
        options=options,
        value=[],  # no default selection
        description="Select Databases:",
        style={"description_width": "initial"},
        layout={"width": "auto"},
    )

    # Button to confirm selection
    confirm_button = widgets.Button(
        description="Connect",
        button_style="success",
        layout={"width": "auto", "margin": "10px 0"},
    )

    # Output widget to display results
    output = widgets.Output()

    def on_button_click(b):
        with output:
            output.clear_output()
            selected = list(select_widget.value)
            if not selected:
                print("No connections selected. Using all connections.")
                selected = connection_names
            print(f"\nConnecting to: {', '.join(selected)}")
            builder.initialize_connections(selected)
            return selected

    confirm_button.on_click(on_button_click)

    # Display widgets
    display(
        widgets.VBox(
            [
                widgets.HTML(value="<h3>Available Database Connections:</h3>"),
                select_widget,
                confirm_button,
                output,
            ]
        )
    )

    return select_widget


def select_tables_widget(builder):
    """Create and display a widget for selecting tables to join.

    Args:
        builder: SmartJoinBuilder instance containing available tables

    Returns:
        dict: Dictionary containing selected tables or None if not enough tables available
    """
    if len(builder.available_tables) < 2:
        print("Not enough tables available for joining. Need at least 2 tables.")
        return None, None

    # Create options for tables
    table_options = [
        (f"{table.connection_name}.{table.database_name}.{table.table_name}", table)
        for table in builder.available_tables
    ]

    # Create dropdown widgets for both tables
    table1_select = widgets.Dropdown(
        options=table_options,
        description="First Table:",
        style={"description_width": "initial"},
        layout={"width": "500px"},  # Make dropdown wider
    )

    table2_select = widgets.Dropdown(
        options=table_options,
        description="Second Table:",
        style={"description_width": "initial"},
        layout={"width": "500px"},  # Make dropdown wider
    )
    use_case_description = widgets.Textarea(
        description="Use Case Description:",
        placeholder='Enter your use case description here... e.g. "I want to find the top 10 products by revenue"',
        style={"description_width": "initial"},
        layout={"width": "500px"},
    )
    # Create output widget for status messages
    output = widgets.Output()

    # Create a container for selected tables
    selected_tables = {"table1": None, "table2": None}

    def on_table1_change(change):
        """Handle changes to the first table selection.

        Args:
            change: Change event object containing the new selection
        """
        selected_tables["table1"] = change.new
        with output:
            output.clear_output()
            if change.new:
                print(
                    f"Selected first table: {change.new.connection_name}.{change.new.database_name}.{change.new.table_name}"
                )

    def on_table2_change(change):
        """Handle changes to the second table selection.

        Args:
            change: Change event object containing the new selection
        """
        selected_tables["table2"] = change.new
        with output:
            output.clear_output()
            if change.new:
                print(
                    f"Selected second table: {change.new.connection_name}.{change.new.database_name}.{change.new.table_name}"
                )

    # Connect the change handlers
    table1_select.observe(on_table1_change, names="value")
    table2_select.observe(on_table2_change, names="value")

    # Create a confirmation button
    confirm_button = widgets.Button(
        description="Confirm Selection",
        button_style="success",
        layout={"width": "auto", "margin": "10px 0"},
    )

    def on_confirm_click(b):
        """Handle confirmation button click event.

        Args:
            b: Button click event object
        """
        with output:
            output.clear_output()
            if selected_tables["table1"] and selected_tables["table2"]:
                print("Selected tables for join:")
                print(
                    f"Table 1: {selected_tables['table1'].connection_name}.{selected_tables['table1'].database_name}.{selected_tables['table1'].table_name}"
                )
                print(
                    f"Table 2: {selected_tables['table2'].connection_name}.{selected_tables['table2'].database_name}.{selected_tables['table2'].table_name}"
                )
                print(f"Use Case Description: {use_case_description.value}")
                print("Initializing join recommendation...")
                join_recommendation = builder.suggest_join(
                    selected_tables["table1"],
                    selected_tables["table2"],
                    use_case_description.value,
                )
                # print(json.dumps(join_recommendation, indent=2))
                # Construct query using DuckDB syntax
                query = builder.construct_duckdb_query(
                    join_recommendation,
                    selected_tables["table1"],
                    selected_tables["table2"],
                )
                print("\n\nGenerated Query:")
                print(query)
            else:
                print("Please select both tables before confirming.")

    confirm_button.on_click(on_confirm_click)

    # Create a container for all widgets
    widgets_container = widgets.VBox(
        [
            widgets.HTML(value="<h3>Select Tables for Join:</h3>"),
            widgets.VBox([table1_select, table2_select], layout={"margin": "10px 0"}),
            widgets.VBox([use_case_description], layout={"margin": "10px 0"}),
            confirm_button,
            output,
        ],
        layout={"padding": "10px", "height": "1000px"},
    )

    display(widgets_container)

    return selected_tables
