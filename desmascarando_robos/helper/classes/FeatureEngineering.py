import logging
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport, compare

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


class FeatureEngineering:
    """
    Classe que realiza a engenharia de features.
    """

    def __init__(self, seed: int = None):
        self.seed = seed

    @staticmethod
    def mode(series):
        return series.mode()[0] if not series.empty else None

    def transform(self, dataframe: pd.DataFrame) -> "FeatureEngineering":
        """
        Feature Engineering

        Parameters:
        - dataframe (pandas.DataFrame): DataFrame a ser inserido

        Returns:
        - X_train, X_test, X_val, y_train, y_test, y_val(pandas.DataFrame): DataFrames
        """
        # Define X e y
        X = dataframe.drop(columns=["resultado"])
        y = dataframe["resultado"]

        X_train_temp, X_test, y_train_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_temp,
            y_train_temp,
            test_size=0.125,
            random_state=self.seed,
            stratify=y_train_temp,
        )

        dataframes = [X_train, X_test, X_val]

        for dataframe in dataframes:
            # Transformando
            dataframe = self.perform_transformations(dataframe)

        return X_train, X_test, X_val, y_train, y_test, y_val

    def perform_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
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

        # Como essas medidas são de tempo irei aplicar transformações temporais
        data["hora_sin"] = np.sin(2 * np.pi * data["hora"] / 24)
        data["hora_cos"] = np.cos(2 * np.pi * data["hora"] / 24)
        data["minuto_sin"] = np.sin(2 * np.pi * data["minuto"] / 60)
        data["minuto_cos"] = np.cos(2 * np.pi * data["minuto"] / 60)
        data["segundo_sin"] = np.sin(2 * np.pi * data["segundo"] / 60)
        data["segundo_cos"] = np.cos(2 * np.pi * data["segundo"] / 60)

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

        # Participante ativo
        data["participante_ativo"] = data["contagem_participante"].apply(
            lambda x: 1 if x > 20 else 0
        )

        # Leilão popular
        data["leilao_popular"] = data["contagem_leilao"].apply(
            lambda x: 1 if x > 50 else 0
        )

        data["periodo_dia"] = np.where(
            data["hora"] < 6,
            "madrugada",
            np.where(
                data["hora"] < 12,
                "manha",
                np.where(data["hora"] < 18, "tarde", "noite"),
            ),
        )

        lances_por_leilao = (
            data.groupby(["id_participante", "leilao"])
            .size()
            .reset_index(name="num_lances")
        )
        estatisticas_lances = (
            lances_por_leilao.groupby("id_participante")["num_lances"]
            .agg(["min", "max", "mean"])
            .reset_index()
        )
        estatisticas_lances = estatisticas_lances.rename(
            columns={
                "min": "min_lances_leilao",
                "max": "max_lances_leilao",
                "mean": "mean_lances_leilao",
            }
        )

        total_leiloes = (
            data.groupby("id_participante")["leilao"]
            .nunique()
            .reset_index(name="total_leiloes")
        )

        estatisticas_tempo = (
            data.groupby("id_participante")["tempo"]
            .agg(["min", "max", "mean"])
            .reset_index()
        )
        estatisticas_tempo = estatisticas_tempo.rename(
            columns={"min": "min_tempo", "max": "max_tempo", "mean": "mean_tempo"}
        )

        mercadoria_stats = (
            data.groupby("id_participante")["mercadoria"]
            .agg(total_mercadorias="nunique", mercadoria_mais_frequente=self.mode)
            .reset_index()
        )

        dispositivo_stats = (
            data.groupby("id_participante")["dispositivo"]
            .agg(dispositivo_mais_usado=self.mode, total_dispositivos="nunique")
            .reset_index()
        )

        pais_stats = (
            data.groupby("id_participante")["pais"]
            .agg(total_paises="nunique", pais_mais_frequente=self.mode)
            .reset_index()
        )

        ip_stats = (
            data.groupby("id_participante")["ip"]
            .agg(total_ips="nunique", ip_mais_frequente=self.mode)
            .reset_index()
        )

        url_stats = (
            data.groupby("id_participante")["url"]
            .agg(total_urls="nunique", url_mais_frequente=self.mode)
            .reset_index()
        )

        resultados = pd.merge(total_leiloes, estatisticas_lances, on="id_participante")
        resultados = pd.merge(resultados, estatisticas_tempo, on="id_participante")
        resultados = pd.merge(resultados, mercadoria_stats, on="id_participante")
        resultados = pd.merge(resultados, dispositivo_stats, on="id_participante")
        resultados = pd.merge(resultados, pais_stats, on="id_participante")
        resultados = pd.merge(resultados, ip_stats, on="id_participante")
        resultados = pd.merge(resultados, url_stats, on="id_participante")

        # Unir as novas features calculadas ao dataset original
        data = data.merge(resultados, on="id_participante", how="left")

        # Removendo colunas que não serão utilizadas
        data = data.drop(columns=["id_participante", "id_lance", "tempo"])

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
        Gera um report

        Parameters:
        - data (pd.DataFrame): Inserir um DataFrame para gerar o report
        - path (str): Path para salvar arquivo
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
        Cria um relatório de comparação entre dois DataFrames.

        Parameters:
        - dataframe1, dataframe2 (pd.DataFrame): DataFrames para comparar.
        - path (str): Path para salvar arquivo
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
        Salva o dataframe em um arquivo parquet.

        Args:
            data (pandas.DataFrame): O dataframe a ser salvo.
            file_path (str): O path do arquivo onde o dataframe será salvo.

        Returns:
            None

        Raises:
            Exception: Se ocorrer um erro ao salvar o arquivo.

        """
        try:
            if os.path.exists(file_path):
                logging.warning("File already exists. It will be overwritten.")

            table = pa.Table.from_pandas(data)
            pq.write_table(table, file_path)
            logging.info("Data successfully saved")
        except Exception as e:
            logging.error(f"Failed to save data: {str(e)}")
