import os

START_DIR = "."
EXTENSIONS = {".py", ".ipynb"}

# Termos-chave que provavelmente aparecem juntos no código
KEYWORDS = ["autoencoder", "confusion", "common"]
SAVE_CALLS = ["savefig", "write_image", "save_fig"]


def find_dynamic_plot_code(start_dir):
    candidates = []

    for root, dirs, files in os.walk(start_dir):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
               and d not in {"venv", ".venv", "__pycache__", "node_modules"}
        ]

        for file in files:
            _, ext = os.path.splitext(file)
            if ext in EXTENSIONS:
                path = os.path.join(root, file)

                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().lower()

                    # Verifica se o arquivo tem salvar figura E menções aos termos
                    has_save = any(call in content for call in SAVE_CALLS)
                    matched_keys = [k for k in KEYWORDS if k in content]

                    # Considera relevante se tiver função de salvar + pelo menos 2 palavras-chave
                    if has_save and len(matched_keys) >= 2:
                        candidates.append((path, matched_keys))

                except Exception as e:
                    print(f"Erro ao ler {path}: {e}")

    return candidates


if __name__ == "__main__":
    results = find_dynamic_plot_code(START_DIR)

    if not results:
        print("Nenhum arquivo correspondente encontrado.")
    else:
        print(f"Encontrado(s) {len(results)} arquivo(s) candidato(s):\n")
        for path, matched_keys in results:
            print(f"Arquivo: {path}")
            print(f"Palavras encontradas: {matched_keys}")
            print("Linhas com salvamento de figura:")

            # Imprime apenas as linhas de savefig para conferência rápida
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for num, line in enumerate(f, start=1):
                    if any(call in line.lower() for call in SAVE_CALLS):
                        print(f"  [L{num}] {line.strip()}")
            print("-" * 70)