from database import PostgresDatabase, BigQueryDatabase, connections
from data_joining import join_data
from metadata import Schema, Databases
postgres_db = PostgresDatabase('postgres_datamancers')
bigquery_db = BigQueryDatabase('bigquery_datamancers')


bigquery_db.connect(fetch_schema=False)
bigquery_data = bigquery_db.fetch_data('SELECT * FROM `airbyte__sourcing.postgres__workers_planned_jobs`')
postgres_db.connect(fetch_schema=False)
postgres_data = postgres_db.fetch_data('SELECT * FROM clients_demo')
joined_data = join_data(bigquery_data, postgres_data, ['id'], 'inner', 'merge')





