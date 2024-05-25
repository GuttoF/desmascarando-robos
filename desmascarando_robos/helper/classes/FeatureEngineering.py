import logging
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ydata_profiling import ProfileReport, compare

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


class FeatureEngineering:
    """
    Classe que realiza a engenharia de features.
    """

    def __init__(self):
        pass

    def transform(self, data: pd.DataFrame) -> "FeatureEngineering":
        # Resolvendo os problemas do alpha2 code em pais
        data["pais"] = data["pais"].replace(
            {
                "Índia": "in",
                "Indonésia": "id",
                "Estados Unidos": "us",
                "Tailândia": "th",
                "Rússia": "ru",
                "Nigéria": "ng",
                "África do Sul": "za",
                "Paraguai": "py",
                "Austrália": "au",
                "Sudão": "sd",
            }
        )

        # Extrair dia
        data["dia"] = data["tempo"].dt.strftime("%d").astype(int)
        # Extrair hora
        data["hora"] = data["tempo"].dt.strftime("%H").astype(int)
        # Extrair minuto
        data["minuto"] = data["tempo"].dt.strftime("%M").astype(int)
        # Extrair segundo
        data["segundo"] = data["tempo"].dt.strftime("%S.%f").astype(float)

        # O endereço IP é uma sequência de números composta por 32 bits (no padrão IPv4).
        # Esse valor consiste em um conjunto de quatro sequências de 8 bits.
        # Cada uma é separada por um ponto e recebe o nome de octeto ou simplesmente byte,
        # pois um byte é formado por 8 bits.
        # Os dois primeiros octetos de um endereço IP identificam a rede
        # e os dois últimos são utilizados na identificação dos dispositivos.
        data["primeiro_octeto_ip"] = data["ip"].apply(lambda x: int(x.split(".")[0]))
        data["segundo_octeto_ip"] = data["ip"].apply(lambda x: int(x.split(".")[1]))

        # Contagem de participantes
        contagem_participante = data["id_participante"].value_counts()
        data["contagem_participante"] = data["id_participante"].map(
            contagem_participante
        )

        # Contagem do leilão
        contagem_leilao = data["leilao"].value_counts()
        data["contagem_leilao"] = data["leilao"].map(contagem_leilao)

        # Contagem conta de pagamento
        contagem_conta_pagamento = data["conta_pagamento"].value_counts()
        data["contagem_conta_pagamento"] = data["conta_pagamento"].map(
            contagem_conta_pagamento
        )

        # Frequência de dispositivos
        frequencia_dispositivo = data["dispositivo"].value_counts()
        data["frequencia_dispositivo"] = data["dispositivo"].map(frequencia_dispositivo)

        # Horário principal
        data["horario_principal"] = data["hora"].apply(
            lambda x: 1 if 9 <= x < 18 else 0
        )

        # As classes de IP são uma forma tradicional de categorizar endereços IP em diferentes faixas, cada uma com um propósito específico.
        # Esta classificação é baseada nos primeiros bits dos endereços IP
        # Classe A: O primeiro octeto vai de 0 a 127 e o uso é para redes muito grandes.
        # Classe B: O primeiro octeto vai de 128 a 191 e o uso é para redes grandes como empresas de médio porte ou universidades.
        # Classe C: O primeiro octeto vai de 192 a 223 e o uso é para redes pequenas como residências ou pequenas empresas.
        # Classe D: O primeiro octeto vai de 224 a 239 e o uso é para multicast.
        # Classe E: O primeiro octeto vai de 240 a 255 e o uso é reservado.
        data["ip_classe"] = np.where(
            data["primeiro_octeto_ip"] <= 127,
            "Classe A",
            np.where(
                data["primeiro_octeto_ip"] <= 191,
                "Classe B",
                np.where(
                    data["primeiro_octeto_ip"] <= 223,
                    "Classe C",
                    np.where(data["primeiro_octeto_ip"] <= 239, "Classe D", "Classe E"),
                ),
            ),
        )

        return data

    def report_na(self, data: pd.DataFrame) -> logging.info:
        """
        Reports the number of missing values (NA) in each column of the given dataframe.

        Args:
            data (pd.DataFrame): The dataframe to check for missing values.

        Returns:
            logging.info: A message indicating the number of missing values in each column.
        """
        if not isinstance(data, pd.DataFrame):
            logging.error("Input is not a pandas DataFrame.")
            return

        for column in data.columns:
            na_count = data[column].isna().sum()
            if na_count > 0:
                return logging.info(
                    f"There are {na_count} NA in the '{column}' column."
                )
        return logging.info("There are no NA values in any column.")

    def get_profile_report(self, data: pd.DataFrame, path: str) -> None:
        """
        Generate a profile report for the given dataframe.

        Parameters:
        - data (pd.DataFrame): Input dataframe to generate the report for.
        - path (str): Path to save the report.
        """
        report = ProfileReport(
            data,
            title="Profile Report",
            correlations={"auto": {"calculate": False}},
        )

        if os.path.exists(path):
            logging.warning("File already exists. It will be overwritten.")
            os.remove(path)

        report.to_file(path)
        logging.info("Profile report saved.")

    def get_comparation_reports(
        self,
        dataframe1: pd.DataFrame,
        dataframe2: pd.DataFrame,
        path: str,
    ) -> None:
        """
        Generate and compare profile reports for training, testing, and validation datasets.

        Parameters:
        - dataframe1, dataframe2 (pd.DataFrame): DataFrames to generate reports for.
        - path (str): Path to save the comparison report.
        """
        dataframes = {"DF1": dataframe1, "DF2": dataframe2}

        reports = {
            name: ProfileReport(
                df, title=name, correlations={"auto": {"calculate": False}}
            )
            for name, df in dataframes.items()
        }
        comparison_report = compare(list(reports.values()))

        if os.path.exists(path):
            logging.warning("File already exists. It will be overwritten.")
            os.remove(path)

        comparison_report.to_file(path)
        logging.info("Comparison report saved.")

    def save_data(self, data: pd.DataFrame, file_path: str) -> None:
        """
        Save the given dataframe to a Parquet file.

        Args:
            data (pandas.DataFrame): The dataframe to be saved.
            file_path (str): The path to the output file.

        Returns:
            None

        Raises:
            Exception: If there is an error while saving the data.

        """
        try:
            if os.path.exists(file_path):
                logging.warning("File already exists. It will be overwritten.")

            table = pa.Table.from_pandas(data)
            pq.write_table(table, file_path)
            logging.info("Data successfully saved")
        except Exception as e:
            logging.error(f"Failed to save data: {str(e)}")
