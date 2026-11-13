"""Treino com PyTorch em vários datasets, comparando paralelismo de dados
(DistributedDataParallel, vários workers) com um processo multithread, em
quantidades diferentes de CPUs e, opcionalmente, com GPU NVIDIA."""

import argparse
import csv
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from data_sources import DATASET_NAMES, load_dataset, prepare_dataset

BATCH_SIZE = 64
SEED = 42
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
RESULTS_DIR = Path("results")
RESULT_FILE_NAME = "results.json"
# Backend de comunicação entre os workers. O gloo funciona na CPU (Windows e
# Linux) e também com tensores na GPU.
DIST_BACKEND = "gloo"
# Memória compartilhada mínima (Linux/Docker) para os workers receberem o
# maior dataset (CIFAR-10, ~0,7 GB em float32) sem copiá-lo.
MIN_SHM_GB = 2
PARALLELISM_MODES = {"data": ["data"], "threads": ["threads"],
                     "both": ["data", "threads"]}
NOT_AVAILABLE = "não disponível"

# Códigos SMBIOS de tipo de memória (Win32_PhysicalMemory.SMBIOSMemoryType).
SMBIOS_MEMORY_TYPES = {20: "DDR", 21: "DDR2", 24: "DDR3", 26: "DDR4",
                       27: "LPDDR", 28: "LPDDR2", 29: "LPDDR3", 30: "LPDDR4",
                       34: "DDR5", 35: "LPDDR5"}


def run_command(command, timeout=15):
    """Executa um comando e devolve a saída, ou None se falhar."""
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def windows_powershell_json(script):
    output = run_command(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         f"{script} | ConvertTo-Json -Compress"]
    )
    if not output:
        return []
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else [data]


def detect_os():
    system = platform.system()
    if system == "Windows":
        rows = windows_powershell_json(
            "Get-CimInstance Win32_OperatingSystem | "
            "Select-Object Caption, Version, BuildNumber"
        )
        if rows:
            row = rows[0]
            return f"{row.get('Caption', '')} {row.get('Version', '')}".strip()
        return f"Windows {platform.release()} {platform.version()}"
    if system == "Linux":
        os_release = read_text("/etc/os-release") or ""
        for line in os_release.splitlines():
            if line.startswith("PRETTY_NAME="):
                name = line.split("=", 1)[1].strip().strip('"')
                return f"{name} (kernel {platform.release()})"
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    return platform.platform()


def detect_cpu_model():
    system = platform.system()
    if system == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif system == "Linux":
        for line in (read_text("/proc/cpuinfo") or "").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    elif system == "Darwin":
        model = run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if model:
            return model
    return platform.processor() or NOT_AVAILABLE


def detect_ram():
    """Devolve (total_gb, velocidade_mhz, tipo)."""
    total_bytes = None
    speed = None
    memory_type = None
    system = platform.system()

    if system == "Windows":
        rows = windows_powershell_json(
            "Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity, "
            "Speed, ConfiguredClockSpeed, SMBIOSMemoryType"
        )
        if rows:
            total_bytes = sum(int(row.get("Capacity") or 0) for row in rows)
            speeds = {
                row.get("ConfiguredClockSpeed") or row.get("Speed")
                for row in rows
            } - {None, 0}
            if speeds:
                speed = "/".join(str(value) for value in sorted(speeds))
            types = {
                SMBIOS_MEMORY_TYPES.get(row.get("SMBIOSMemoryType"))
                for row in rows
            } - {None}
            if types:
                memory_type = "/".join(sorted(types))
    elif system == "Linux":
        for line in (read_text("/proc/meminfo") or "").splitlines():
            if line.startswith("MemTotal:"):
                total_bytes = int(line.split()[1]) * 1024
        # Velocidade só via dmidecode, que exige root e acesso ao hardware
        # (normalmente indisponível dentro de containers Docker).
        if shutil.which("dmidecode"):
            output = run_command(["dmidecode", "-t", "memory"]) or ""
            speeds = set()
            types = set()
            for line in output.splitlines():
                line = line.strip()
                if line.startswith(("Configured Memory Speed:", "Speed:")):
                    value = line.split(":", 1)[1].split()[0]
                    if value.isdigit():
                        speeds.add(int(value))
                elif line.startswith("Type:"):
                    value = line.split(":", 1)[1].strip()
                    if value not in ("Unknown", "Other"):
                        types.add(value)
            if speeds:
                speed = "/".join(str(value) for value in sorted(speeds))
            if types:
                memory_type = "/".join(sorted(types))
    elif system == "Darwin":
        output = run_command(["sysctl", "-n", "hw.memsize"])
        if output and output.isdigit():
            total_bytes = int(output)

    total_gb = round(total_bytes / 1024**3, 2) if total_bytes else NOT_AVAILABLE
    return total_gb, speed or NOT_AVAILABLE, memory_type or NOT_AVAILABLE


def detect_gpu_model():
    if torch.version.cuda and torch.version.hip is None:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    output = run_command(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    return output.splitlines()[0].strip() if output else None


def collect_system_info():
    """Informações do computador onde o treino roda (coletadas uma vez)."""
    ram_total_gb, ram_speed, ram_type = detect_ram()
    return {
        "sistema_operacional": detect_os(),
        "em_docker": Path("/.dockerenv").exists(),
        "python_versao": platform.python_version(),
        "pytorch_versao": torch.__version__,
        "cpu_modelo": detect_cpu_model(),
        "cpu_nucleos_logicos": os.cpu_count(),
        "gpu_modelo_nvidia": detect_gpu_model() or NOT_AVAILABLE,
        "ram_total_gb": ram_total_gb,
        "ram_velocidade_mhz": ram_speed,
        "ram_tipo": ram_type,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Modelos (um por tipo de dado)
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Imagens e espectrogramas. O pooling adaptativo leva qualquer tamanho de
    entrada para 7x7 (no MNIST 28x28 o modelo é idêntico ao original)."""

    def __init__(self, in_channels, num_outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(self.adaptive_pool(x), 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TextCNN(nn.Module):
    """Texto como sequência de índices de palavras."""

    def __init__(self, vocab_size, num_outputs, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(embedding_dim, 128, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)        # (N, emb, seq)
        x = F.relu(self.conv(x)).amax(dim=2)         # max pooling global
        return self.fc(self.dropout(x))


class TabularMLP(nn.Module):
    def __init__(self, in_features, num_outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)


class SequenceLSTM(nn.Module):
    def __init__(self, in_features, num_outputs, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


MODELS = {
    "imagem": SimpleCNN,
    "texto": TextCNN,
    "tabela": TabularMLP,
    "serie_temporal": SequenceLSTM,
}


def build_model(bundle):
    return MODELS[bundle.kind](
        num_outputs=bundle.num_outputs, **bundle.model_params
    )


# ---------------------------------------------------------------------------
# Treino e avaliação
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, device, task, optimizer=None):
    """Uma passada pelo loader. Treina se ``optimizer`` for informado.

    Devolve as somas (loss, métrica, nº de amostras) para que, no paralelismo
    de dados, os valores de todos os workers possam ser somados (all-reduce).
    A métrica é o nº de acertos (classificação) ou a soma dos erros absolutos
    (regressão).
    """
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    metric_sum = 0.0
    total = 0

    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            if task == "regressao":
                outputs = outputs.squeeze(1)
            loss = criterion(outputs, targets)

            if training:
                optimizer.zero_grad()
                # No DDP, o backward também sincroniza (all-reduce) os
                # gradientes entre os workers.
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            loss_sum += loss.item() * batch_size
            if task == "regressao":
                metric_sum += (outputs - targets).abs().sum().item()
            else:
                metric_sum += (outputs.argmax(dim=1) == targets).sum().item()
            total += batch_size

    return loss_sum, metric_sum, total


def combine(sums, task, distributed):
    """Soma os resultados de todos os workers e devolve (loss média,
    métrica): acurácia (%) na classificação ou MAE na regressão."""
    values = torch.tensor(sums, dtype=torch.float64)
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    loss_sum, metric_sum, total = values.tolist()
    metric = metric_sum / total
    if task == "classificacao":
        metric *= 100
    return loss_sum / total, metric


def save_plot(values, title, ylabel, path):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(values) + 1), values, marker="o")
    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def train_worker(rank, config):
    """Treina em um worker.

    - Paralelismo de dados (``workers`` > 1): cada worker é um processo, com
      uma cópia do modelo (DistributedDataParallel) e 1 thread. O
      DistributedSampler dá a cada worker uma parte diferente dos dados de
      treino, e os gradientes são sincronizados a cada passo.
    - Threads (``workers`` == 1): um único processo, com ``threads`` threads
      dividindo cada operação.

    Só o worker 0 (rank 0) imprime e salva os resultados.
    """
    bundle = config["bundle"]
    workers = config["workers"]
    threads = config["threads"]
    device = torch.device(config["device"])
    output_dir = config["output_dir"]
    distributed = workers > 1
    is_main = rank == 0
    uses_gpu = device.type == "cuda"
    is_regression = bundle.task == "regressao"
    metric_name = "MAE" if is_regression else "Acc"

    torch.set_num_threads(threads)
    if distributed:
        # Processos filhos terminam sem esvaziar o buffer da saída; com
        # line_buffering, cada linha impressa aparece na hora.
        sys.stdout.reconfigure(line_buffering=True)
        dist.init_process_group(
            backend=DIST_BACKEND, init_method=config["init_method"],
            rank=rank, world_size=workers,
        )
    set_seed(SEED)

    if distributed:
        train_sampler = DistributedSampler(
            bundle.train, num_replicas=workers, rank=rank, shuffle=True,
            seed=SEED,
        )
        train_loader = DataLoader(bundle.train, batch_size=BATCH_SIZE,
                                  sampler=train_sampler)
        # Avaliação também dividida entre os workers (sem repetir amostras).
        test_part = Subset(bundle.test,
                           range(rank, len(bundle.test), workers))
        test_loader = DataLoader(test_part, batch_size=BATCH_SIZE)
    else:
        train_sampler = None
        train_loader = DataLoader(bundle.train, batch_size=BATCH_SIZE,
                                  shuffle=True)
        test_loader = DataLoader(bundle.test, batch_size=BATCH_SIZE)

    model = build_model(bundle).to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[device.index or 0] if uses_gpu else None
        )

    if is_main:
        print(f"Workers (processos): {workers} | threads por worker: "
              f"{torch.get_num_threads()}")
        print("Dispositivo:", device)
        if uses_gpu:
            print("GPU:", torch.cuda.get_device_name(device))
        print(f"Amostras de treino: {len(bundle.train)} "
              f"(~{len(bundle.train) // workers} por worker)")
        print(f"Amostras de teste: {len(bundle.test)}")
        print(f"Formato de uma amostra: {tuple(bundle.train[0][0].shape)}")
        print(f"Batch por worker: {BATCH_SIZE} | batch global: "
              f"{BATCH_SIZE * workers}")
        print(f"Batches por época, por worker: {len(train_loader)}")
        print(f"Modelo: {build_model(bundle).__class__.__name__} "
              f"({total_params:,} parâmetros)")

    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses, train_metrics = [], []
    test_losses, test_metrics = [], []

    # Todos os workers começam juntos; o tempo de criação dos processos não
    # entra na medição.
    if distributed:
        dist.barrier()
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)   # embaralha diferente a cada época
        train_loss, train_metric = combine(
            run_epoch(model, train_loader, criterion, device, bundle.task,
                      optimizer),
            bundle.task, distributed,
        )
        test_loss, test_metric = combine(
            run_epoch(model, test_loader, criterion, device, bundle.task),
            bundle.task, distributed,
        )
        train_losses.append(train_loss)
        train_metrics.append(train_metric)
        test_losses.append(test_loss)
        test_metrics.append(test_metric)

        if is_main:
            unit = "" if is_regression else "%"
            print(
                f"Época [{epoch + 1}/{NUM_EPOCHS}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train {metric_name}: {train_metric:.4f}{unit} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test {metric_name}: {test_metric:.4f}{unit}"
            )

    if uses_gpu:
        torch.cuda.synchronize(device)
    if distributed:
        dist.barrier()
    total_time = time.time() - start_time
    if distributed:
        dist.destroy_process_group()
    if not is_main:
        return None

    print(f"\nTempo total de execução: {total_time:.2f} segundos")
    results = {
        "modo": "data_parallel_ddp" if config["parallelism"] == "data"
        else "processo_unico_threads",
        "paralelismo": config["parallelism"],
        "dataset": bundle.name,
        "tipo_dado": bundle.kind,
        "tarefa": bundle.task,
        "descricao_dataset": bundle.description,
        "fonte_dados": bundle.source,
        "amostras_treino": len(bundle.train),
        "amostras_teste": len(bundle.test),
        "modelo": build_model(bundle).__class__.__name__,
        "parametros_modelo": total_params,
        "epochs": NUM_EPOCHS,
        "cpus": config["cpus"],
        "workers": workers,
        "threads_por_worker": threads,
        "backend_distribuido": DIST_BACKEND if distributed else None,
        "batch_por_worker": BATCH_SIZE,
        "batch_global": BATCH_SIZE * workers,
        "learning_rate": LEARNING_RATE,
        "device": str(device),
        "gpu": uses_gpu,
        "gpu_nome": torch.cuda.get_device_name(device) if uses_gpu else None,
        "tempo_total_segundos": round(total_time, 2),
        "loss_final_treino": train_losses[-1],
        "loss_final_teste": test_losses[-1],
        # Classificação: acurácia (%). Regressão: MAE, na unidade do alvo.
        "acc_final_treino": None if is_regression else train_metrics[-1],
        "acc_final_teste": None if is_regression else test_metrics[-1],
        "mae_final_treino": train_metrics[-1] if is_regression else None,
        "mae_final_teste": test_metrics[-1] if is_regression else None,
        "seed": SEED,
        "data_hora_execucao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **config["system_info"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / RESULT_FILE_NAME
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    print(f"Resultados salvos em: {output_file}")

    save_plot(train_losses, f"Loss de Treinamento por Época ({bundle.name})",
              "Loss", output_dir / "training_loss.png")
    if is_regression:
        save_plot(test_metrics, f"MAE no Conjunto de Teste ({bundle.name})",
                  "MAE", output_dir / "test_mae.png")
    else:
        save_plot(test_metrics,
                  f"Acurácia no Conjunto de Teste ({bundle.name})",
                  "Acurácia (%)", output_dir / "test_accuracy.png")
    print(f"Gráficos salvos em: {output_dir}")
    return results


def run_training(bundle, cpus, device, parallelism, output_dir, system_info):
    """Um treino com ``cpus`` CPUs no modo de paralelismo escolhido."""
    if parallelism == "data":
        workers, threads = cpus, 1
        label = f"paralelismo de dados, {cpus} worker(s) x 1 thread"
    else:
        workers, threads = 1, cpus
        label = f"1 processo x {cpus} thread(s)"
    gpu_label = "com GPU" if device.type == "cuda" else "sem GPU"
    print(f"\n===== {bundle.name}: {label}, {gpu_label} =====")

    config = {
        "bundle": bundle,
        "cpus": cpus,
        "workers": workers,
        "threads": threads,
        "parallelism": parallelism,
        "device": str(device),
        "output_dir": output_dir,
        "system_info": system_info,
    }
    if workers == 1:
        return train_worker(0, config)

    # Paralelismo de dados: um processo por worker. Os tensores do dataset são
    # passados em memória compartilhada (não são copiados para cada processo).
    # A sincronização entre os processos usa um arquivo temporário.
    store = Path(tempfile.gettempdir()) / f"ddp_store_{uuid.uuid4().hex}"
    config["init_method"] = store.as_uri()
    (output_dir / RESULT_FILE_NAME).unlink(missing_ok=True)
    try:
        mp.spawn(train_worker, args=(config,), nprocs=workers, join=True)
    finally:
        store.unlink(missing_ok=True)
    with (output_dir / RESULT_FILE_NAME).open(encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------

def parse_bool(value):
    normalized = value.strip().lower()
    if normalized in ("true", "1", "sim", "s", "yes", "y"):
        return True
    if normalized in ("false", "0", "nao", "não", "n", "no"):
        return False
    raise argparse.ArgumentTypeError(
        f"valor inválido {value!r}; use true ou false."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina e avalia modelos PyTorch em um ou mais datasets, "
        "comparando quantidades de CPUs e, opcionalmente, GPU NVIDIA."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset(s) a treinar, separados por vírgula, ou 'all' para "
        "todos, um após o outro. Opções: " + ", ".join(DATASET_NAMES) + ".",
    )
    parser.add_argument(
        "train_csv",
        type=Path,
        nargs="?",
        help="(Opcional, só para o dataset mnist) CSV de treino: rótulo "
        "seguido de 784 pixels, sem cabeçalho. Sem ele, o MNIST é baixado.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        help="(Opcional, só para o dataset mnist) CSV de teste no mesmo "
        "formato. Obrigatório se o CSV de treino for informado.",
    )
    parser.add_argument(
        "--cpu-amount",
        help="(Obrigatório, exceto com --download-only) Quantidades de CPUs "
        "(threads do PyTorch), separadas por vírgula. Cada valor gera um "
        "treino: '4' treina só com 4 CPUs; '1,2,4' treina com 1, depois 2 e "
        "depois 4.",
    )
    parser.add_argument(
        "--parallelism",
        choices=list(PARALLELISM_MODES),
        default="data",
        help="data (padrão): paralelismo de dados, com N workers (processos) "
        "de 1 CPU cada, via DistributedDataParallel. threads: 1 processo com "
        "N threads. both: roda os dois, para comparar. N = cada valor de "
        "--cpu-amount.",
    )
    parser.add_argument(
        "--use-gpu",
        type=parse_bool,
        default=False,
        metavar="{true,false}",
        help="Se true, cada quantidade de CPUs é treinada duas vezes: com GPU "
        "NVIDIA (CUDA) e sem GPU, para comparar. Se false (padrão), treina só "
        "na CPU.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Pasta onde os datasets baixados ficam guardados (padrão: data). "
        "Cada dataset é baixado só uma vez.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Só baixa e prepara os datasets em --data-dir, sem treinar. Use "
        "antes do docker build para colocar os dados dentro da imagem.",
    )
    args = parser.parse_args()

    requested = [name.strip().lower() for name in args.dataset.split(",")]
    if "all" in requested:
        if requested != ["all"]:
            parser.error("use --dataset all sozinho, sem outros nomes.")
        args.dataset_list = list(DATASET_NAMES)
    else:
        unknown = [name for name in requested if name not in DATASET_NAMES]
        if unknown:
            parser.error(
                f"dataset(s) desconhecido(s): {', '.join(unknown)}. Opções: "
                + ", ".join(DATASET_NAMES) + " ou all."
            )
        if len(set(requested)) != len(requested):
            parser.error(f"--dataset tem valores repetidos: {args.dataset!r}")
        args.dataset_list = requested

    if args.cpu_amount is None:
        if not args.download_only:
            parser.error("o argumento --cpu-amount é obrigatório.")
        args.cpu_amount = "1"   # não é usado: --download-only não treina
    try:
        args.cpu_list = [int(v) for v in args.cpu_amount.split(",")]
    except ValueError:
        args.cpu_list = []
    if not args.cpu_list or min(args.cpu_list) < 1:
        parser.error(
            "--cpu-amount deve ser uma lista de inteiros positivos separados "
            f"por vírgula (ex.: 4 ou 1,2,4); recebido: {args.cpu_amount!r}"
        )
    if len(set(args.cpu_list)) != len(args.cpu_list):
        parser.error(f"--cpu-amount tem valores repetidos: {args.cpu_amount!r}")

    # No Linux, os workers compartilham o dataset via /dev/shm. No Docker ele
    # tem só 64 MB por padrão, e o treino falharia no meio da execução.
    uses_workers = ("data" in PARALLELISM_MODES[args.parallelism]
                    and max(args.cpu_list) > 1)
    if uses_workers and not args.download_only and Path("/dev/shm").is_dir():
        stats = os.statvfs("/dev/shm")
        shm_gb = stats.f_bavail * stats.f_frsize / 1024**3
        if shm_gb < MIN_SHM_GB:
            parser.error(
                f"o paralelismo de dados precisa de pelo menos {MIN_SHM_GB} GB "
                f"de memória compartilhada (/dev/shm), mas há {shm_gb:.2f} GB. "
                "No Docker, adicione --shm-size=4g ao docker run."
            )

    if args.download_only and args.use_gpu:
        parser.error("--download-only não treina; não use com --use-gpu true.")

    if (args.train_csv is None) != (args.test_csv is None):
        parser.error(
            "para usar CSVs próprios do MNIST informe os dois: o CSV de treino "
            "e --test-csv."
        )
    if args.train_csv is not None:
        if "mnist" not in args.dataset_list:
            parser.error("os CSVs só são usados com o dataset mnist.")
        args.train_csv = args.train_csv.resolve()
        args.test_csv = args.test_csv.resolve()
        if not args.train_csv.is_file():
            parser.error(f"arquivo de treino não encontrado: {args.train_csv}")
        if not args.test_csv.is_file():
            parser.error(f"arquivo de teste não encontrado: {args.test_csv}")

    # Somente GPUs NVIDIA (CUDA). O PyTorch para AMD (ROCm) também responde
    # torch.cuda.is_available(), por isso torch.version.hip é verificado.
    if args.use_gpu and (
        torch.version.cuda is None
        or torch.version.hip is not None
        or not torch.cuda.is_available()
    ):
        parser.error(
            "--use-gpu true, mas nenhuma GPU NVIDIA (CUDA) está disponível "
            "para o PyTorch. Verifique os drivers NVIDIA, a instalação do "
            "PyTorch com CUDA e, no Docker, a opção --gpus all."
        )
    return args


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def write_results_table(all_results, path):
    """Tabela com todos os treinos (uma linha por treino), fácil de abrir no
    Excel/Planilhas. Separador ";" e decimais com vírgula para o Excel em
    português; UTF-8 com BOM para os acentos."""
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(all_results[0].keys()), delimiter=";"
        )
        writer.writeheader()
        for result in all_results:
            writer.writerow({
                key: str(value).replace(".", ",")
                if isinstance(value, float) else value
                for key, value in result.items()
            })


def format_metric(result):
    if result["tarefa"] == "regressao":
        return f"MAE de teste {result['mae_final_teste']:.4f}"
    return f"acurácia de teste {result['acc_final_teste']:.2f}%"


def main():
    args = parse_args()
    available = os.cpu_count() or 1
    if max(args.cpu_list) > available:
        print(
            f"Aviso: {max(args.cpu_list)} CPUs solicitadas, mas apenas "
            f"{available} visíveis; o desempenho não deve escalar além disso."
        )

    system_info = collect_system_info()
    print("Informações do sistema:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    print("Datasets:", ", ".join(args.dataset_list))

    devices = [torch.device("cpu")]
    if args.use_gpu:
        devices.insert(0, torch.device("cuda"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table_file = RESULTS_DIR / "all_results.csv"
    summary_file = RESULTS_DIR / "all_results.json"
    all_results = []
    failures = []

    # Fase 1: baixa e prepara todos os datasets antes de qualquer treino.
    # Assim, os treinos não ficam esperando downloads no meio da execução.
    print("\n########## Fase 1: download e preparação dos datasets ##########")
    for dataset_name in args.dataset_list:
        print(f"\n--- Preparando {dataset_name} ---")
        try:
            prepare_dataset(dataset_name, args.data_dir,
                            use_csv=args.train_csv is not None)
            print(f"{dataset_name}: pronto.")
        except Exception as error:  # noqa: BLE001 - segue para o próximo
            if len(args.dataset_list) == 1:
                raise
            print(f"Erro ao preparar {dataset_name}: {error}. Pulando.")
            failures.append(dataset_name)
    ready = [name for name in args.dataset_list if name not in failures]
    print("\nDatasets prontos para treino:", ", ".join(ready) or "nenhum")
    if args.download_only:
        if failures:
            raise SystemExit(
                "Falha ao preparar: " + ", ".join(failures)
                + ". Rode de novo para continuar os downloads."
            )
        print(f"Download concluído. Dados em: {args.data_dir.resolve()}")
        return

    # Fase 2: treinos.
    print("\n########## Fase 2: treinos ##########")
    for dataset_name in ready:
        print(f"\n########## Dataset: {dataset_name} ##########")
        try:
            bundle = load_dataset(
                dataset_name, args.data_dir,
                train_csv=args.train_csv, test_csv=args.test_csv,
            )
        except Exception as error:  # noqa: BLE001 - segue para o próximo
            # Com vários datasets, uma falha (ex.: sem internet) não
            # interrompe os demais.
            if len(args.dataset_list) == 1:
                raise
            print(f"Erro ao carregar {dataset_name}: {error}. Pulando.")
            failures.append(dataset_name)
            continue

        for cpus in args.cpu_list:
            for device in devices:
                for parallelism in PARALLELISM_MODES[args.parallelism]:
                    prefix = "workers" if parallelism == "data" else "threads"
                    suffix = "_gpu" if device.type == "cuda" else ""
                    output_dir = (RESULTS_DIR / dataset_name
                                  / f"{prefix}_{cpus}{suffix}")
                    all_results.append(run_training(
                        bundle, cpus, device, parallelism, output_dir,
                        system_info,
                    ))
                    # Salva a cada treino, para não perder nada se algo parar.
                    write_results_table(all_results, table_file)
                    with summary_file.open("w", encoding="utf-8") as file:
                        json.dump(all_results, file, indent=4,
                                  ensure_ascii=False)
        del bundle

    if not all_results:
        raise SystemExit("Nenhum treino foi concluído.")

    print("\nResumo:")
    for result in all_results:
        print(
            f"  {result['dataset']:<16} {result['cpus']:>2} CPU(s) "
            f"{result['paralelismo']:<7} "
            f"{'com GPU' if result['gpu'] else 'sem GPU'}: "
            f"{result['tempo_total_segundos']:.2f}s | {format_metric(result)}"
        )
    if failures:
        print("Datasets que falharam (download ou carregamento):", ", ".join(failures))
    print(f"Tabela de resultados salva em: {table_file}")
    print(f"Resumo (JSON) salvo em: {summary_file}")


if __name__ == "__main__":
    main()
