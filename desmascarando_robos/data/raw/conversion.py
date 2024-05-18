import os
import sys

import duckdb
from dotenv import load_dotenv

# Load the environment variables
env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=env_path)
path = os.getenv("HOMEPATH")

if path is None:
    print("The HOMEPATH environment variable is not set.")
    sys.exit(1)

# Paths
df_train_path = path + "/data/raw/train.csv"
df_test_path = path + "/data/raw/test.csv"
df_lances_path = path + "/data/raw/lances.csv"
db_path = path + "/data/interim/data.db"

for paths in [df_train_path, df_test_path, df_lances_path]:
    if not os.path.exists(paths):
        print(f"The file located in {paths} does not exist.")
        sys.exit(1)

try:
    # Read CSV
    df_train = duckdb.read_csv(df_train_path)
    df_test = duckdb.read_csv(df_test_path)
    df_lances = duckdb.read_csv(df_lances_path)

    # Create the connection
    conn = duckdb.connect(db_path)
    # Create the tables
    conn.execute("CREATE TABLE lances AS SELECT * FROM df_lances")
    conn.execute("CREATE TABLE train AS SELECT * FROM df_train")
    conn.execute("CREATE TABLE test AS SELECT * FROM df_test")
except Exception as e:
    print(f"An error occurred while creating the tables: {e}")
    sys.exit(1)
finally:
    conn.close()
