#!/usr/bin/env python3
"""
Pipeline orquestrador para rodar modelos localmente via LM Studio.

Pré-requisitos:
    - LM Studio instalado com os modelos baixados
    - Servidor local do LM Studio rodando (aba "Local Server")

Uso:
    python3 pipeline.py
"""

import subprocess
import sys
import os
import time
import requests

# ─────────────────────────────────────────────
# CONFIGURAÇÃO — edite aqui
# ─────────────────────────────────────────────

MODELS = [
    "qwen/qwen3.5-9b"
]
    # "google/gemma-4-e4b",
    # "mistralai/ministral-3-14b-reasoning",
# Endereço do servidor local do LM Studio
LM_STUDIO_URL = "http://localhost:1234"

# Caminho para o seu script principal
MAIN_SCRIPT = "main.py"

# Tempo de espera após carregar um modelo (segundos)
# Aumente se seu computador for mais lento para carregar
LOAD_WAIT = 10

# ─────────────────────────────────────────────
# FUNÇÕES
# ─────────────────────────────────────────────

def check_server():
    """Verifica se o servidor do LM Studio está rodando."""
    try:
        r = requests.get(f"{LM_STUDIO_URL}/v1/models", timeout=5)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def load_model(model_name):
    """Carrega um modelo no LM Studio via API."""
    print(f"Carregando modelo '{model_name}'...")
    r = requests.post(
        f"{LM_STUDIO_URL}/api/v1/models/load",
        json={"model": model_name},
        timeout=300,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"Falha ao carregar '{model_name}': {r.status_code} — {r.text}"
        )
    print(f"Modelo carregado. Aguardando {LOAD_WAIT}s para estabilizar...")
    time.sleep(LOAD_WAIT)


def unload_model(model_name):
    """Descarrega o modelo da memória."""
    print(f"Descarregando '{model_name}'...")
    requests.post(
        f"{LM_STUDIO_URL}/api/v1/models/unload",
        json={"instance_id": model_name},
        timeout=60,
    )


def run_model(model_name, index, total):
    """Executa main.py para o modelo carregado."""
    print(f"\n{'─' * 50}")
    print(f"[{index}/{total}] {model_name}")
    print(f"{'─' * 50}")

    env = os.environ.copy()
    env["LM_STUDIO_URL"] = LM_STUDIO_URL

    start = time.time()
    result = subprocess.run(
        [sys.executable, MAIN_SCRIPT, "--model", model_name],
        env=env,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"Falha no modelo '{model_name}' "
            f"(exit code {result.returncode}) após {elapsed:.1f}s."
        )

    print(f"Concluído em {elapsed:.1f}s.")


def validate_setup():
    if not os.path.isfile(MAIN_SCRIPT):
        print(f"Script '{MAIN_SCRIPT}' não encontrado no diretório atual.")
        sys.exit(1)

    if not MODELS:
        print("A lista MODELS está vazia.")
        sys.exit(1)

    print("Verificando servidor do LM Studio...")
    if not check_server():
        print(f"Servidor do LM Studio não encontrado em {LM_STUDIO_URL}")
        print("   Abra o LM Studio e inicie o servidor na aba 'Local Server'.")
        sys.exit(1)
    print("Servidor encontrado.")


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  PIPELINE DE MODELOS — LM STUDIO")
    print(f"  {len(MODELS)} modelos na fila")
    print("=" * 50)

    validate_setup()

    total = len(MODELS)
    pipeline_start = time.time()

    for i, model in enumerate(MODELS, start=1):
        try:
            load_model(model)
            run_model(model, i, total)
        except RuntimeError as e:
            print(f"\n{'=' * 50}")
            print("  PIPELINE INTERROMPIDO — ERRO")
            print("=" * 50)
            print(str(e))
            print(f"  Concluídos: {i - 1}/{total}")
            print(f"  Pendentes:  {MODELS[i - 1:]}")
            print("=" * 50)
            unload_model(model)
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nPipeline cancelado pelo usuário.")
            unload_model(model)
            sys.exit(130)

        unload_model(model)

    elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 50}")
    print(f"PIPELINE CONCLUÍDO — {total} modelos em {elapsed:.1f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()