import subprocess
import sys
import os
import time


# ─────────────────────────────────────────────
# CONFIGURAÇÃO — edite aqui
# ─────────────────────────────────────────────

MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
]

# Caminho para o seu script principal
MAIN_SCRIPT = "main.py"

# ─────────────────────────────────────────────
# FUNÇÕES
# ─────────────────────────────────────────────

def validate_setup():
    errors = []

    if not os.path.isfile(MAIN_SCRIPT):
        errors.append(f"Script '{MAIN_SCRIPT}' não encontrado no diretório atual.")

    if not MODELS:
        errors.append("A lista MODELS está vazia.")

    if errors:
        for e in errors:
            print(f"{e}")
        sys.exit(1)


def run_model(model_name, index, total):
    print(f"\n{'─' * 50}")
    print(f"[{index}/{total}] {model_name}")
    print(f"{'─' * 50}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, MAIN_SCRIPT, "--model", model_name],
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"Falha no modelo '{model_name}' "
            f"(exit code {result.returncode}) após {elapsed:.1f}s."
        )

    print(f"Concluído em {elapsed:.1f}s.")


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  PIPELINE DE MODELOS — GROQ")
    print(f"  {len(MODELS)} modelos na fila")
    print("=" * 50)

    validate_setup()

    total = len(MODELS)
    pipeline_start = time.time()

    for i, model in enumerate(MODELS, start=1):
        try:
            run_model(model, i, total)
        except RuntimeError as e:
            print(f"\n{'=' * 50}")
            print("  PIPELINE INTERROMPIDO — ERRO")
            print("=" * 50)
            print(str(e))
            print(f"  Concluídos: {i - 1}/{total}")
            print(f"  Pendentes:  {MODELS[i - 1:]}")
            print("=" * 50)
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nPipeline cancelado pelo usuário.")
            sys.exit(130)


    elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 50}")
    print(f"  PIPELINE CONCLUÍDO — {total} modelos em {elapsed:.1f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()