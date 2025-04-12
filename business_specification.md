# Case Study

## Background
Your colleagues, BI analysts, often need to query data from 2 different database systems. They need to query data from multiple tables in different databases (read-only), join both results into one, and retrieve the resulting data set for their ad-hoc analysis.

## Objective
The objective of this case study is to design and present a solution that enables users to write queries to 2 different databases (Postgres, MySQL, SQLite, Snowflake, BigQuery, etc...choose 2 that you like and have access to) and define join columns and output columns. No additional operations on output columns are expected.

## Tasks
- [x] Develop a Python code to query different database systems.
- [x] Add a function to join results from both systems.
- [x] Handle duplicate columns.
- [x] Prepare a testing script for queries and join definition - show more complex queries.
- [ ] Call the LLM of your choice that will provide metadata for the resulting table using context information from source tables (table names, columns, comments if available).
- [ ] Alternatively, you can use LLM/AI agent to create the join from human input. In that case, input will just define what tables to join and the agent will decide how to join them.
- [x] Log important information in useful form for analytical purposes (like average query time etc...).
- [x] Suggest how to securely handle passwords for DB connections for this script.

## Output
Provide a functioning .py script or Jupyter notebook, which we can test during the 2nd round of interviews. The code should have at least basic comments describing what it does. We prefer object-oriented programming, but you can use functional programming or combine both approaches.