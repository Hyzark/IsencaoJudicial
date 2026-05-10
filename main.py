import os
import sys
import traceback
from pathlib import Path

# Garante que o diretório do projeto está no path para os imports funcionarem
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import DOWNLOADS_DIR, MODEL_PATH, RESULTS_EVASAO_DIR, RESULTS_ISENCAO_DIR, RESULTS_EIXO_DIR, ensure_parent

# Import das tuas classes (ajusta os nomes se necessário conforme os teus ficheiros)
try:
    from app_pdf_ecovias import ValidadorPDFEcovias
    from app_excel_eixovp import ValidadorExcelEixoVP
    from app_excel_rota_bandeiras import ValidadorExcelBandeiras
    from app_pdf_autoban import ValidadorPassagens as ValidadorAutoban
    from app_pdf_intervias import ValidadorPDFNovoFormato as ValidadorIntervias
    from app_excel_colinas import ValidadorNovoFormato as ValidadorColinas
    from app_excel_spvias import ValidadorExcelPro as ValidadorSPvias
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    input("Prime Enter para sair...")
    sys.exit(1)

def menu():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*60)
        print("      SISTEMA UNIFICADO DE VALIDAÇÃO DE PASSAGENS (ARTESP)")
        print("="*60)
        print("1. Ecovias - Evasão (PDF)")
        print("2. ViaPaulista - Eixo Suspenso (Excel)")
        print("3. Rota das Bandeiras - Isenção Judicial (Excel)")
        print("4. AutoBan - Isenção Judicial (PDF)")
        print("5. Intervias/Arteris - Evasão (PDF)")
        print("6. Colinas - Evasão (Excel)")
        print("7. SPvias - Evasão (Excel)")
        print("0. Sair")
        print("-"*60)

        escolha = input("Escolha uma opção: ")

        if escolha == '0':
            break

        processar_opcao(escolha)
        input("\nProcessamento concluído. Prime Enter para voltar ao menu...")


def executar_validador(opcao, caminho_entrada):
    """
    Instancia o validador correto e chama processar().

    caminho_entrada pode ser:
      - um arquivo único  (modo 1 do sub-menu)
      - uma pasta         (modo 2 do sub-menu)

    Validadores cujo processar() só aceita pasta (spvias/colinas) recebem
    o pai do arquivo quando a entrada for um arquivo único.
    """
    caminho_entrada = Path(caminho_entrada)
    nome_base = caminho_entrada.stem if caminho_entrada.is_file() else caminho_entrada.name

    print(f"\n-> Processando: {caminho_entrada.name}")
    try:
        if opcao == '1':
            saida = ensure_parent(RESULTS_EVASAO_DIR / f"resultado_ecovias_{nome_base}.xlsx")
            v = ValidadorPDFEcovias(yolo_weights_path=str(MODEL_PATH))
            v.processar(str(caminho_entrada), str(saida))

        elif opcao == '2':
            saida = ensure_parent(RESULTS_EIXO_DIR / f"resultado_eixo_{nome_base}.xlsx")
            v = ValidadorExcelEixoVP(yolo_weights_path=str(MODEL_PATH))
            v.processar(str(caminho_entrada), str(saida))

        elif opcao == '3':
            saida = ensure_parent(RESULTS_ISENCAO_DIR / f"resultado_bandeiras_{nome_base}.xlsx")
            v = ValidadorExcelBandeiras(yolo_weights_path=str(MODEL_PATH))
            v.processar(str(caminho_entrada), str(saida))

        elif opcao == '4':
            saida = ensure_parent(RESULTS_EVASAO_DIR / f"resultado_autoban_{nome_base}.xlsx")
            v = ValidadorAutoban(yolo_weights_path=str(MODEL_PATH))
            v.processar(str(caminho_entrada), str(saida))

        elif opcao == '5':
            saida = ensure_parent(RESULTS_ISENCAO_DIR / f"resultado_intervias_{nome_base}.xlsx")
            v = ValidadorIntervias(yolo_weights_path=str(MODEL_PATH))
            v.processar(str(caminho_entrada), str(saida))

        elif opcao == '6':
            # ValidadorColinas.processar() só aceita pasta — se vier um arquivo
            # único, usa a pasta pai para não quebrar o glob interno.
            entrada_colinas = (
                caminho_entrada.parent
                if caminho_entrada.is_file()
                else caminho_entrada
            )
            saida = ensure_parent(RESULTS_EIXO_DIR / f"resultado_colinas_{nome_base}.xlsx")
            v = ValidadorColinas(yolo_weights_path=str(MODEL_PATH))
            v.processar(str(entrada_colinas), str(saida))

        elif opcao == '7':
            # ValidadorExcelPro.processar() só aceita pasta — mesmo tratamento do Colinas.
            entrada_spvias = (
                caminho_entrada.parent
                if caminho_entrada.is_file()
                else caminho_entrada
            )
            saida = ensure_parent(RESULTS_EVASAO_DIR / f"resultado_spvias_{nome_base}.xlsx")
            v = ValidadorSPvias(yolo_weights_path=str(MODEL_PATH))
            v.processar(str(entrada_spvias), str(saida))

    except Exception as e:
        print(f"Erro ao processar {caminho_entrada.name}: {e}")
        traceback.print_exc()


def processar_opcao(opcao):
    """Sub-menu para escolha entre arquivo único ou lote."""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*60)
        print("                MODO DE PROCESSAMENTO")
        print("="*60)
        print("1. Processar um ÚNICO arquivo")
        print("2. Processar TODOS os arquivos de uma pasta")
        print("0. VOLTAR ao Menu Principal")

        sub_opcao = input("\nSelecione uma opção: ").strip()

        if sub_opcao == '0':
            return

        elif sub_opcao == '1':
            # ---- LÓGICA DE ARQUIVO ÚNICO ----
            arquivos = list(DOWNLOADS_DIR.glob("*.*"))
            if not arquivos:
                print(f"\nNenhum arquivo encontrado em: {DOWNLOADS_DIR}")
                input("Pressione Enter para voltar...")
                continue

            print("\nArquivos disponíveis nos Downloads:")
            print("0. VOLTAR")
            for i, arq in enumerate(arquivos):
                print(f"{i+1}. {arq.name}")

            escolha_arq = input("\nSelecione o número do arquivo (ou 0 para voltar): ").strip()
            if escolha_arq == '0':
                continue

            try:
                idx = int(escolha_arq) - 1
                if 0 <= idx < len(arquivos):
                    executar_validador(opcao, arquivos[idx])
                    input("\nProcessamento concluído! Pressione Enter para voltar ao menu...")
                    return
                else:
                    print("Opção inválida!")
                    input("Pressione Enter para tentar novamente...")
            except ValueError:
                print("Por favor, digite um número válido.")
                input("Pressione Enter para tentar novamente...")

        elif sub_opcao == '2':
            # ---- LÓGICA DE PASTA INTEIRA (LOTE) ----
            print(f"\nPasta padrão: {DOWNLOADS_DIR}")
            caminho_pasta = input("Digite o caminho da pasta (ou Enter para usar a padrão, 0 para voltar): ").strip()

            if caminho_pasta == '0':
                continue

            pasta_alvo = Path(caminho_pasta) if caminho_pasta else DOWNLOADS_DIR

            if not pasta_alvo.exists() or not pasta_alvo.is_dir():
                print(f"\nA pasta '{pasta_alvo}' não foi encontrada.")
                input("Pressione Enter para tentar novamente...")
                continue

            arquivos_pasta = [f for f in pasta_alvo.iterdir() if f.is_file()]
            if not arquivos_pasta:
                print(f"\nA pasta '{pasta_alvo}' está vazia.")
                input("Pressione Enter para tentar novamente...")
                continue

            print(f"\nEncontrados {len(arquivos_pasta)} arquivos. Iniciando processamento em lote...")
            # Passa a PASTA directamente — o validador itera internamente
            executar_validador(opcao, pasta_alvo)

            print("\nProcessamento em lote concluído com sucesso!")
            input("Pressione Enter para voltar ao menu principal...")
            return

        else:
            print("Opção inválida.")
            input("Pressione Enter para tentar novamente...")


if __name__ == "__main__":
    menu()