import pandas as pd
import numpy as np


def mode(series):
        return series.mode()[0] if not series.empty else None

def perform_transformations(data: pd.DataFrame) -> pd.DataFrame:
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
        "A",
        np.where(
            data["primeiro_octeto_ip"] <= 191,
            "B",
            np.where(
                data["primeiro_octeto_ip"] <= 223,
                "C",
                np.where(data["primeiro_octeto_ip"] <= 239, "D", "E"),
            ),
        ),
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

    estatisticas_hora_cos = (
        data.groupby("id_participante")["hora_cos"]
        .agg(["min", "max", "mean"])
        .reset_index()
    )
    estatisticas_hora_cos = estatisticas_hora_cos.rename(
        columns={
            "min": "min_hora_cos",
            "max": "max_hora_cos",
            "mean": "mean_hora_cos",
        }
    )

    estatisticas_minuto_sin = (
        data.groupby("id_participante")["minuto_sin"]
        .agg(["min", "max", "mean"])
        .reset_index()
    )
    estatisticas_minuto_sin = estatisticas_minuto_sin.rename(
        columns={
            "min": "min_minuto_sin",
            "max": "max_minuto_sin",
            "mean": "mean_minuto_sin",
        }
    )

    estatisticas_minuto_cos = (
        data.groupby("id_participante")["minuto_cos"]
        .agg(["min", "max", "mean"])
        .reset_index()
    )
    estatisticas_minuto_cos = estatisticas_minuto_cos.rename(
        columns={
            "min": "min_minuto_cos",
            "max": "max_minuto_cos",
            "mean": "mean_minuto_cos",
        }
    )

    estatisticas_segundo_sin = (
        data.groupby("id_participante")["segundo_sin"]
        .agg(["min", "max", "mean"])
        .reset_index()
    )
    estatisticas_segundo_sin = estatisticas_segundo_sin.rename(
        columns={
            "min": "min_segundo_sin",
            "max": "max_segundo_sin",
            "mean": "mean_segundo_sin",
        }
    )

    estatisticas_segundo_cos = (
        data.groupby("id_participante")["segundo_cos"]
        .agg(["min", "max", "mean"])
        .reset_index()
    )
    estatisticas_segundo_cos = estatisticas_segundo_cos.rename(
        columns={
            "min": "min_segundo_cos",
            "max": "max_segundo_cos",
            "mean": "mean_segundo_cos",
        }
    )

    primeiro_octeto_stats = (
        data.groupby("id_participante")["primeiro_octeto_ip"]
        .agg(primeiro_octeto="nunique", primeiro_octeto_mais_frequente=mode)
        .reset_index()
    )

    segundo_octeto_stats = (
        data.groupby("id_participante")["segundo_octeto_ip"]
        .agg(segundo_octeto="nunique", segundo_octeto_mais_frequente=mode)
        .reset_index()
    )

    horario_principal_stats = (
        data.groupby("id_participante")["horario_principal"]
        .agg(
            horario_principal="nunique", horario_principal_mais_frequente=mode
        )
        .reset_index()
    )

    ip_class_stats = (
        data.groupby("id_participante")["ip_classe"]
        .agg(ip_class="nunique", ip_class_mais_frequente=mode)
        .reset_index()
    )

    periodo_dia = (
        data.groupby("id_participante")["periodo_dia"]
        .agg(periodo_dia="nunique", periodo_dia_mais_frequente=mode)
        .reset_index()
    )

    mercadoria_stats = (
        data.groupby("id_participante")["mercadoria"]
        .agg(total_mercadorias="nunique", mercadoria_mais_frequente=mode)
        .reset_index()
    )

    dispositivo_stats = (
        data.groupby("id_participante")["dispositivo"]
        .agg(dispositivo_mais_usado=mode, total_dispositivos="nunique")
        .reset_index()
    )

    pais_stats = (
        data.groupby("id_participante")["pais"]
        .agg(total_paises="nunique", pais_mais_frequente=mode)
        .reset_index()
    )

    ip_stats = (
        data.groupby("id_participante")["ip"]
        .agg(total_ips="nunique", ip_mais_frequente=mode)
        .reset_index()
    )

    url_stats = (
        data.groupby("id_participante")["url"]
        .agg(total_urls="nunique", url_mais_frequente=mode)
        .reset_index()
    )

    cols_to_drop = ["dia", "hora", "minuto", "segundo"]
    data = data.drop(columns=cols_to_drop)

    resultados = pd.merge(total_leiloes, estatisticas_lances, on="id_participante")
    resultados = pd.merge(resultados, estatisticas_tempo, on="id_participante")
    resultados = pd.merge(resultados, estatisticas_hora_cos, on="id_participante")
    resultados = pd.merge(resultados, estatisticas_minuto_sin, on="id_participante")
    resultados = pd.merge(resultados, estatisticas_minuto_cos, on="id_participante")
    resultados = pd.merge(
        resultados, estatisticas_segundo_sin, on="id_participante"
    )
    resultados = pd.merge(
        resultados, estatisticas_segundo_cos, on="id_participante"
    )
    resultados = pd.merge(resultados, primeiro_octeto_stats, on="id_participante")
    resultados = pd.merge(resultados, segundo_octeto_stats, on="id_participante")
    resultados = pd.merge(resultados, horario_principal_stats, on="id_participante")
    resultados = pd.merge(resultados, ip_class_stats, on="id_participante")
    resultados = pd.merge(resultados, periodo_dia, on="id_participante")
    resultados = pd.merge(resultados, mercadoria_stats, on="id_participante")
    resultados = pd.merge(resultados, dispositivo_stats, on="id_participante")
    resultados = pd.merge(resultados, pais_stats, on="id_participante")
    resultados = pd.merge(resultados, ip_stats, on="id_participante")
    resultados = pd.merge(resultados, url_stats, on="id_participante")

    # Unir as novas features calculadas ao dataset original
    data = data.merge(resultados, on="id_participante", how="left")

    return data