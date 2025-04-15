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
- [x] Alternatively, you can use LLM/AI agent to create the join from human input. In that case, input will just define what tables to join and the agent will decide how to join them.
- [x] Log important information in useful form for analytical purposes (like average query time etc...).
- [x] Suggest how to securely handle passwords for DB connections for this script.


## Output
Provide a functioning .py script or Jupyter notebook, which we can test during the 2nd round of interviews. The code should have at least basic comments describing what it does. We prefer object-oriented programming, but you can use functional programming or combine both approaches.


### TODOs:

- udělat všude komentáře + logy, aby bylo jasné co zrovna apka dělá


# Poznámky k prezentaci
1. zadání jsem místy pojal poměrně freestylově, kdy jsem si řekl, co by se do tohoto use case hodilo. V některých bodech bych měl otázky, ale spíš jsem chtěl vytvořit na základě zadání rychlej prototyp
2. na začátku jsem měl skromější verzi, ale pak se mi to poměrně nafouklo, 
3. používal jsem cursor, se kterým jsem konzultoval jak design, tak ho nechal generovat výstupy
### jak jsem postupoval:
1. propojení dvou database systémů: BigQuery a Postgres na mých interních datech (stará databáze pro pet project monitorování práce a fakturace)
2. původně jsem vytvořil jednoduché python funkce, ale pak jsem si uvědomil, že by to šlo řešit přes duckdb, které umožňuje integrace na oba dva systémy -> myšlenka vytvořit "federated query engine" kdy by se BI analytici mohli v rámci jedné query  dotazovat na oba systémy, duck db by na pozadí stáhlo data ze systémů a zároveň by vytvořilo finální výstup. BI analytici by tak nemuseli psát víc než jednu query, což je fajn pro analýzy (ale ne moc praktické pro nějaké automatizace).
3. dále jsem postavil recommender na joiny, který na základě zadaných tabulek vytáhne sample tabulek a ai následně doporučuje jednu z strategií joinu (left,inner,full outer,(šlo by přidat UNION)), je možné také rovnou spustit exekuci joinu. AI table description část jsem vynechal.
4. to celé jsem zabalil do jupyter notebooku, který se tváří téměř jako UI, a kdy všechna backend logika je v rámci importovaných modulů (v produkci by se dal tento modul normálně importovat jako git repozitář)
5. uživatel v `config.yaml` nastavuje connection konfiguraci na n databází, zároveň se zde nastavují další parametry pro produkci (kam ukládat metadata atd.)
6. na pozadí se také cachují metadata o databázi (`data/metadata.yaml`) (názvy tabulek,sloupců atd.), které se využívají pro UI generování listů, případně může analytik/AI využít pro přehled o metadatech v tabulkách (trošku vnímám jako obdobu dbt model yaml properties)
7. logging napříč celou appkou, pro debugging.
8. také nastavený logging query performance (kdy se vytahují jak perf data přímo z databází, ale také se sleduje query perf přímo v duckdb)

### ad security: 
1. přihlašovací údaje do databází: uloženy v konfiguraci `config.yaml`, odkud si je aplikace tahá.
2. openapi klíč uložen v .env file
3. pro ještě větší zabezpečení by šlo využít a do apky naimplementovat secret manager (například GCP secret manager), kdy by se přístup (např k OPENAI API klíči) přiděloval na základě credentials uživatele v platformě.