import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────

RESULTS_DIR = '../results'
PLOTS_DIR   = '../results/plots'
CONFIGS     = ['baseline', 'soft', 'aggressive']
METRICAS    = ['jaccard_medio', 'perplexidade', 'self_bleu', 'distinct_1', 'distinct_2']

# ─────────────────────────────────────────────
# CARREGAR DADOS
# ─────────────────────────────────────────────

def carregar_dados():
    """
    Lê todos os CSVs em RESULTS_DIR e retorna um DataFrame
    com coluna extra 'modelo' e 'config' extraídas do nome do arquivo.
    Padrão esperado: resultados_{modelo}_{config}.csv
    """
    arquivos = glob.glob(os.path.join(RESULTS_DIR, 'resultados_*.csv'))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV encontrado em '{RESULTS_DIR}'.")

    dfs = []
    for path in arquivos:
        nome = os.path.basename(path).replace('resultados_', '').replace('.csv', '')

        # Extrai config (última parte) e modelo (resto)
        for config in CONFIGS:
            if nome.endswith(f'_{config}'):
                modelo = nome[: -(len(config) + 1)]
                break
        else:
            print(f"Aviso: arquivo ignorado (padrão não reconhecido): {path}")
            continue

        df = pd.read_csv(path)
        df['modelo'] = modelo
        df['config'] = config
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────────
# GERAR BOXPLOTS
# ─────────────────────────────────────────────

def gerar_boxplots(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    modelos = sorted(df['modelo'].unique())
    total = len(modelos) * len(METRICAS)
    gerados = 0

    for modelo in modelos:
        df_modelo = df[df['modelo'] == modelo]

        for metrica in METRICAS:
            if metrica not in df_modelo.columns:
                print(f"Aviso: métrica '{metrica}' não encontrada para {modelo}.")
                continue

            # Monta lista de valores por config (uma por caixa no boxplot)
            dados = [
                df_modelo[df_modelo['config'] == config][metrica].dropna().values
                for config in CONFIGS
            ]

            # Ignora se todos os grupos estiverem vazios
            if all(len(d) == 0 for d in dados):
                print(f"Aviso: sem dados para {modelo} / {metrica}.")
                continue

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.boxplot(dados, labels=CONFIGS, patch_artist=True,
                       boxprops=dict(facecolor='#d0e4f7', color='#2b6cb0'),
                       medianprops=dict(color='#c0392b', linewidth=2),
                       whiskerprops=dict(color='#2b6cb0'),
                       capprops=dict(color='#2b6cb0'),
                       flierprops=dict(marker='o', color='#999', markersize=4))

            ax.set_title(f'{modelo}\n{metrica}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Configuração')
            ax.set_ylabel(metrica)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

            plt.tight_layout()

            filename = f"{modelo}_{metrica}.png"
            filepath = os.path.join(PLOTS_DIR, filename)
            plt.savefig(filepath, dpi=150)
            plt.close()

            gerados += 1
            print(f"[{gerados}/{total}] Salvo: {filepath}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Carregando dados...")
    df = carregar_dados()
    print(f"Dados carregados: {len(df)} linhas, {df['modelo'].nunique()} modelo(s).\n")
    gerar_boxplots(df)
    print("\nConcluído.")