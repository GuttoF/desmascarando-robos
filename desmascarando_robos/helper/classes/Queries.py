import duckdb


class DuckQueries:
    """
    A class that helps in building SQL queries in duckdb.

    Methods:
    - reset: Resets the query string.
    - select: Adds the SELECT clause to the query.
    - get_connection: Establishes a connection to the database.
    - from_table: Adds the FROM clause to the query.
    - where: Adds the WHERE clause to the query.
    - join: Adds a JOIN clause to the query.
    - group_by: Adds the GROUP BY clause to the query.
    - limit: Adds the LIMIT clause to the query.
    - column_names: Adds a custom clause to retrieve column names from a table.
    - column_types: Adds a custom clause to retrieve column names and data types from a table.
    - drop_column: Removes a specified column from the selection.
    - concat_tables: Concatenates two tables.
    - as_table: Treats a subquery as a table for subsequent operations.
    - build: Builds the final query string and resets the query.

    Usage:
    ```
    query_builder = DuckQueries()
    query = query_builder.select("column1, column2").from_table("table").where("condition").build()
    ```
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the query string.
        """
        self.query = ""
        self.selected_columns = "*"
        self.is_subquery = False

    def get_connection(self, conn_path: str) -> "duckdb.Connection":
        """
        Establishes a connection to the specified database.

        Args:
            conn_path (str): The path to the database file.

        Returns:
            duckdb.Connection: The connection object.

        Raises:
            Exception: If an error occurs while establishing the connection.
        """
        try:
            return duckdb.connect(database=conn_path, read_only=False)
        except Exception as e:
            raise Exception(f"Failed to connect to the database: {e}")

    def select(self, args: str) -> "DuckQueries":
        """
        Adds the SELECT clause to the query.

        Args:
        - args: A string representing the columns to select.

        Returns:
        - self: The DuckQueries instance.
        """
        self.selected_columns = args
        return self

    def from_table(self, table: str) -> "DuckQueries":
        """
        Adds the FROM clause to the query.

        Args:
        - table: A string representing the table name.

        Returns:
        - self: The DuckQueries instance.
        """
        if self.is_subquery:
            self.query += f" FROM ({table})"
            self.is_subquery = False
        else:
            self.query += f" FROM {table}"
        return self

    def where(self, condition: str) -> "DuckQueries":
        """
        Adds the WHERE clause to the query.

        Args:
        - condition: A string representing the condition.

        Returns:
        - self: The DuckQueries instance.
        """
        self.query += f" WHERE {condition}"
        return self

    def join(
        self, table: str, condition: str, join_type: str = "INNER"
    ) -> "DuckQueries":
        """
        Adds a JOIN clause to the query.

        Args:
        - table: A string representing the table name to join.
        - condition: A string representing the join condition.
        - join_type: A string representing the type of join (INNER, LEFT, RIGHT, FULL). Default is INNER.

        Returns:
        - self: The DuckQueries instance.
        """
        self.query += f" {join_type} JOIN {table} ON {condition}"
        return self

    def group_by(self, args: str) -> "DuckQueries":
        """
        Adds the GROUP BY clause to the query.

        Args:
        - args: A string representing the columns to group by.

        Returns:
        - self: The DuckQueries instance.
        """
        self.query += f" GROUP BY {args}"
        return self

    def limit(self, n: int) -> "DuckQueries":
        """
        Adds the LIMIT clause to the query.

        Args:
        - n: An integer representing the limit.

        Returns:
        - self: The DuckQueries instance.
        """
        self.query += f" LIMIT {n}"
        return self

    def shape(self, table: str) -> "DuckQueries":
        """
        Retrieves the number of rows and columns of a table.

        Args:
            table (str): The table name.

        Returns:
            self: The DuckQueries instance.
        """
        self.query = f"SELECT (SELECT COUNT(*) FROM {table}) AS row_count, (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}') AS column_count"
        return self

    def column_names(self, table: str) -> "DuckQueries":
        """
        Adds a custom clause to retrieve column names from a table.

        Args:
        - table: A string representing the table name.

        Returns:
        - self: The DuckQueries instance.
        """
        self.query += f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}'"
        return self

    def count_nulls_from_table(self, table: str) -> "DuckQueries":
        """
        Constructs a query to count null values across all columns in the specified table.

        Args:
            table (str): The table name.

        Returns:
            self: The DuckQueries instance.
        """
        self.query = "SELECT "
        subquery = f"(SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}')"
        self.query += f"(SELECT STRING_AGG('COUNT(CASE WHEN ' || COLUMN_NAME || ' IS NULL THEN 1 END) AS ' || COLUMN_NAME, ', ') FROM {subquery}) AS query_part;"
        return self

    def count_zeros_from_table(self, table: str) -> "DuckQueries":
        """
        Constructs a query to count zero values across all columns of specific types (BIGINT or DOUBLE) in the specified table.

        Args:
            table (str): The table name.

        Returns:
            self: The DuckQueries instance.
        """
        self.query = "SELECT "
        subquery = f"(SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}' AND DATA_TYPE IN ('BIGINT', 'DOUBLE'))"
        self.query += f"(SELECT STRING_AGG('COUNT(CASE WHEN ' || COLUMN_NAME || ' = 0 THEN 1 END) AS ' || COLUMN_NAME, ', ') FROM {subquery}) AS query_part;"
        return self

    def column_types_from_table(self, table: str) -> "DuckQueries":
        """
        Retrieves column names and their data types from a table.

        Args:
            table (str): The table name.

        Returns:
            self: The DuckQueries instance.
        """
        self.query += f"SELECT * FROM {table}; "
        self.query += f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}';"
        return self

    def drop_column(self, column: str) -> "DuckQueries":
        """
        Removes a specified column from the selection.

        Args:
        - column: A string representing the column name to drop.

        Returns:
        - self: The DuckQueries instance.
        """
        columns = self.selected_columns.split(",")
        columns = [col.strip() for col in columns if col.strip() != column]
        self.selected_columns = ", ".join(columns)
        return self

    def concat_tables(self, table1: str, table2: str) -> "DuckQueries":
        """
        Concatenates two tables.

        Args:
        - table1: A string representing the first table name.
        - table2: A string representing the second table name.

        Returns:
        - self: The DuckQueries instance.
        """
        self.query = f"SELECT * FROM {table1} UNION ALL SELECT * FROM {table2}"
        self.is_subquery = True
        return self

    def as_table(self) -> str:
        """
        Treats the current query as a subquery and returns the query string.

        Returns:
        - query: The current query string.
        """
        query = f"({self.query})"
        self.reset()
        return query

    def build(self) -> str:
        """
        Builds the final query string and resets the query.

        Returns:
        - final_query: The final query string.
        """
        if self.is_subquery:
            final_query = self.query + ";"
        else:
            final_query = (
                f"SELECT {self.selected_columns}{self.query};"
                if self.selected_columns != "*"
                else self.query + ";"
            )
        self.reset()
        return final_query
