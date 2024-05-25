import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
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

final_train_path = path + "/data/raw/train.feather"
final_test_path = path + "/data/raw/test.feather"
final_lances_path = path + "/data/raw/lances.feather"

# Loading the data
df_train = pd.read_csv(df_train_path)
df_test = pd.read_csv(df_test_path)
df_lances = pd.read_csv(df_lances_path)

# Converting to Table
table_train = pa.Table.from_pandas(df_train)
table_test = pa.Table.from_pandas(df_test)
table_lances = pa.Table.from_pandas(df_lances)

# Saving the data
if not os.path.exists(final_train_path):
    feather.write_feather(table_train, final_train_path)
    feather.write_feather(table_test, final_test_path)
    feather.write_feather(table_lances, final_lances_path)
else:
    sys.exit(1)
