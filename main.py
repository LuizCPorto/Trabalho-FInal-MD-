import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools import tool


load_dotenv()

DATA_PATH = "PORTES_2025.csv"

PORTES_DF: Optional[pd.DataFrame] = None


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Carrega o CSV PORTES_2025 em memória (uma única vez) e faz
    pequenos tratamentos.
    """
    global PORTES_DF
    if PORTES_DF is None:
        df = pd.read_csv(path, encoding="latin1", sep=";")

        df.columns = [c.strip().upper() for c in df.columns]

        if "TOTAL" in df.columns:
            df["TOTAL"] = pd.to_numeric(df["TOTAL"], errors="coerce").fillna(0)

        PORTES_DF = df

    return PORTES_DF


@tool
def consultar_portes(
    uf: Optional[str] = None,
    municipio: Optional[str] = None,
    sexo: Optional[str] = None,
    especie_arma: Optional[str] = None,
    marca_arma: Optional[str] = None,
    calibre_arma: Optional[str] = None,
    status: Optional[str] = None,
    tipo: Optional[str] = None,
    abrangencia: Optional[str] = None,
    ano_emissao: Optional[int] = None,
    mes_missao: Optional[int] = None,
    agrupar_por: Optional[str] = None,
    top_n: Optional[int] = 5,
) -> str:
    """
    Consulta o arquivo PORTES_2025 com base em filtros opcionais.

    Esta função deve ser usada pelo agente para responder perguntas do usuário
    sobre o dataset de Portes de Armas de Fogo - Defesa Pessoal.

    Parâmetros (todos opcionais):
    - uf: sigla do estado (ex.: "BA")
    - municipio: nome do município (ex.: "SALVADOR")
    - sexo: "M" ou "F"
    - especie_arma: ex.: "Pistola"
    - marca_arma: ex.: "TAURUS ARMAS S.A."
    - calibre_arma: ex.: ".380 ACP"
    - status: ex.: "Ativo - Válido", "Cancelado", "Vencido"
    - tipo: ex.: "Defesa Pessoal", "Funcional"
    - abrangencia: ex.: "Estadual", "Nacional"
    - ano_emissao: ano numérico (ex.: 2025)
    - mes_missao: mês numérico (1 a 12)
    - agrupar_por: nome de uma coluna para gerar ranking
                   (ex.: "UF", "MUNICIPIO", "CALIBRE_ARMA", "MARCA_ARMA")
    - top_n: quantidade de linhas no ranking (padrão = 5)

    Retorna:
    - Um resumo em texto com o total de portes e, se solicitado,
      um ranking agregado pela coluna escolhida.
    """
    df = load_dataset()

    filtro = df

    def filtrar_coluna_texto(frame: pd.DataFrame, coluna: str, valor: str) -> pd.DataFrame:
        coluna = coluna.upper()
        if coluna not in frame.columns:
            return frame
        serie = frame[coluna].astype(str).str.upper()
        return frame[serie.str.contains(valor.upper(), na=False)]

    if uf:
        filtro = filtrar_coluna_texto(filtro, "UF", uf)

    if municipio:
        filtro = filtrar_coluna_texto(filtro, "MUNICIPIO", municipio)

    if sexo:
        filtro = filtrar_coluna_texto(filtro, "SEXO", sexo)

    if especie_arma:
        filtro = filtrar_coluna_texto(filtro, "ESPECIE_ARMA", especie_arma)

    if marca_arma:
        filtro = filtrar_coluna_texto(filtro, "MARCA_ARMA", marca_arma)

    if calibre_arma:
        filtro = filtrar_coluna_texto(filtro, "CALIBRE_ARMA", calibre_arma)

    if status:
        filtro = filtrar_coluna_texto(filtro, "STATUS", status)

    if tipo:
        filtro = filtrar_coluna_texto(filtro, "TIPO", tipo)

    if abrangencia:
        filtro = filtrar_coluna_texto(filtro, "ABRANGENCIA", abrangencia)

    if ano_emissao is not None and "ANO_EMISSAO" in filtro.columns:
        filtro = filtro[filtro["ANO_EMISSAO"] == int(ano_emissao)]

    if mes_missao is not None and "MES_MISSAO" in filtro.columns:
        filtro = filtro[filtro["MES_MISSAO"] == int(mes_missao)]

    if filtro.empty:
        return "Nenhum registro encontrado com esses filtros."

    if "TOTAL" in filtro.columns:
        total_portes = int(filtro["TOTAL"].sum())
    else:
        total_portes = len(filtro)

    if not agrupar_por:
        return (
            f"Foram encontrados {total_portes} portes de arma com os filtros informados.\n"
            f"(Registros: {len(filtro)})"
        )

    col = agrupar_por.upper()
    if col not in filtro.columns:
        colunas_disponiveis = ", ".join(filtro.columns)
        return (
            f"A coluna '{agrupar_por}' não existe no dataset.\n"
            f"Colunas disponíveis: {colunas_disponiveis}"
        )

    if "TOTAL" in filtro.columns:
        tabela = (
            filtro.groupby(col)["TOTAL"]
            .sum()
            .reset_index()
            .sort_values("TOTAL", ascending=False)
        )
    else:
        tabela = (
            filtro.groupby(col)
            .size()
            .reset_index(name="TOTAL")
            .sort_values("TOTAL", ascending=False)
        )

    if top_n is not None:
        tabela = tabela.head(int(top_n))

    linhas = []
    for _, row in tabela.iterrows():
        chave = str(row[col])
        valor = int(row["TOTAL"])
        linhas.append(f"- {chave}: {valor} portes")

    ranking = "\n".join(linhas)

    return (
        f"Foram encontrados {total_portes} portes de arma com os filtros informados.\n"
        f"Ranking por {col} (top {len(tabela)}):\n"
        f"{ranking}"
    )


def criar_agente() -> Agent:
    """
    Cria o agente que usa o modelo Gemini e a tool consultar_portes.
    """
    model = Gemini(id="gemini-2.0-flash")

    agent = Agent(
        model=model,
        name="AgentePortes",
        description=(
            "Você é um chatbot especializado no dataset PORTES_2025, "
            "que contém portes de armas de fogo na categoria Defesa Pessoal. "
            "Você deve responder em português claro, explicando o contexto. "
            "Sempre que precisar de números ou estatísticas, use a ferramenta "
            "'consultar_portes' para consultar o CSV."
        ),
        tools=[consultar_portes],
        markdown=False,
    )
    return agent


def main():
    try:
        load_dataset()
    except FileNotFoundError:
        print(f"Erro: não encontrei o arquivo {DATA_PATH}. "
              f"Verifique o caminho no código.")
        return

    if not os.getenv("GOOGLE_API_KEY"):
        print("Erro: variável de ambiente GOOGLE_API_KEY não definida.")
        print("Crie um arquivo .env com GOOGLE_API_KEY=... ou exporte a variável.")
        return

    agent = criar_agente()

    print("=== Chatbot de Portes de Armas de Fogo (PORTES_2025) ===")
    print("Faça perguntas sobre dados o porte de Armas de Fogo no ano de 2025.")
    print("Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("Você: ").strip()
        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Chatbot: Até mais!")
            break

        try:
            agent.print_response(pergunta, stream=True)
            print()
        except Exception as e:
            print(f"[ERRO AO CHAMAR O AGENTE] {e}")


if __name__ == "__main__":
    main()
