"""Download e preparação dos datasets usados no treino.

Os arquivos são baixados das mesmas fontes oficiais usadas por
``tensorflow.keras.datasets``, ``tensorflow_datasets`` e ``sklearn.datasets``,
sem depender do TensorFlow. Tudo fica em cache na pasta de dados (``--data-dir``),
então cada dataset é baixado só uma vez.
"""

import csv
import gzip
import io
import pickle
import tarfile
import time
import urllib.request
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

KERAS_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
SPEECH_COMMANDS_URL = (
    "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
)
JENA_URL = KERAS_URL + "jena_climate_2009_2016.csv.zip"

SEED = 42
CSV_COLUMN_COUNT = 1 + 28 * 28
DOWNLOAD_TIMEOUT = 60      # segundos sem receber dados antes de desistir
DOWNLOAD_ATTEMPTS = 5

# Texto
IMDB_VOCAB_SIZE = 10_000
IMDB_MAX_LEN = 256
PAD_INDEX = 0
OOV_INDEX = 1

# Áudio: 10 comandos clássicos do Speech Commands (mesma tarefa do TFDS).
SPEECH_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off",
                "stop", "go"]
SAMPLE_RATE = 16_000
N_FFT = 480        # janela de 30 ms
HOP_LENGTH = 320   # passo de 20 ms
N_MELS = 40

# Série temporal: amostra por hora, usa 5 dias de histórico para prever a
# temperatura 24 h depois (mesma ideia do exemplo oficial do Keras).
JENA_STEP = 6            # 1 linha a cada 10 min -> 1 a cada hora
JENA_PAST_HOURS = 120
JENA_FUTURE_HOURS = 24
JENA_TEMPERATURE_COLUMN = "T (degC)"


@dataclass
class DatasetBundle:
    """Dataset pronto para treino, com o que é preciso para montar o modelo."""

    name: str
    task: str                 # "classificacao" ou "regressao"
    kind: str                 # "imagem", "texto", "tabela", "serie_temporal"
    train: Dataset
    test: Dataset
    source: str
    num_outputs: int          # nº de classes (classificação) ou 1 (regressão)
    model_params: dict = field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def download(url, destination):
    """Baixa ``url`` para ``destination`` (se ainda não existir)."""
    destination = Path(destination)
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    print(f"Baixando {url}")
    last_error = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            _download_attempt(url, partial)
            partial.replace(destination)
            return destination
        except OSError as error:   # inclui timeout e conexão interrompida
            last_error = error
            print(f"  Falha no download (tentativa {attempt}/"
                  f"{DOWNLOAD_ATTEMPTS}): {error}. Tentando de novo...")
            time.sleep(min(30, 5 * attempt))
    raise OSError(
        f"não foi possível baixar {url} após {DOWNLOAD_ATTEMPTS} tentativas: "
        f"{last_error}"
    )


def _download_attempt(url, partial):
    """Baixa para ``partial``, continuando de onde parou se o arquivo
    incompleto já existir e o servidor aceitar (cabeçalho Range)."""
    done = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "dataset-downloader"}
    if done:
        headers["Range"] = f"bytes={done}-"
    request = urllib.request.Request(url, headers=headers)
    # O timeout evita que uma conexão parada trave o script para sempre.
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
        if done and response.status == 206:
            mode = "ab"
            print(f"  Continuando de {done / 1024**2:.0f} MB")
        else:
            mode, done = "wb", 0
        total = done + int(response.headers.get("Content-Length") or 0)
        last_step = -1
        with partial.open(mode) as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                done += len(chunk)
                if total:
                    step = done * 10 // total
                    if step != last_step:
                        print(f"  {done * 100 // total}% de "
                              f"{total / 1024**2:.0f} MB")
                    last_step = step
        if total and done < total:
            raise OSError(f"download incompleto ({done} de {total} bytes)")


def standardize_images(train_images, test_images):
    """Normaliza (N, C, H, W) por canal com a média/desvio do treino."""
    mean = train_images.mean(dim=(0, 2, 3), keepdim=True)
    std = train_images.std(dim=(0, 2, 3), keepdim=True)
    return (train_images - mean) / std, (test_images - mean) / std


def image_bundle(name, source, description, train_x, train_y, test_x, test_y,
                 num_classes):
    """``*_x`` em uint8 com formato (N, C, H, W)."""
    train_x = torch.from_numpy(train_x).float().div_(255.0)
    test_x = torch.from_numpy(test_x).float().div_(255.0)
    train_x, test_x = standardize_images(train_x, test_x)
    return DatasetBundle(
        name=name,
        task="classificacao",
        kind="imagem",
        train=TensorDataset(train_x, torch.as_tensor(train_y, dtype=torch.long)),
        test=TensorDataset(test_x, torch.as_tensor(test_y, dtype=torch.long)),
        source=source,
        num_outputs=num_classes,
        model_params={"in_channels": train_x.shape[1]},
        description=description,
    )


def split_indices(n, test_fraction, labels=None):
    """Divide 0..n-1 em treino/teste de forma reprodutível (estratificada se
    ``labels`` for informado)."""
    rng = np.random.default_rng(SEED)
    if labels is None:
        order = rng.permutation(n)
        cut = int(round(n * (1 - test_fraction)))
        return order[:cut], order[cut:]
    train_parts, test_parts = [], []
    for value in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == value))
        cut = int(round(len(idx) * (1 - test_fraction)))
        train_parts.append(idx[:cut])
        test_parts.append(idx[cut:])
    return np.concatenate(train_parts), np.concatenate(test_parts)


def tabular_bundle(name, source, description, x, y, task, num_outputs,
                   stratify):
    train_idx, test_idx = split_indices(
        len(x), 0.2, labels=y if stratify else None
    )
    x = x.astype(np.float32)
    mean = x[train_idx].mean(axis=0)
    std = x[train_idx].std(axis=0)
    std[std == 0] = 1.0
    x = (x - mean) / std
    y_tensor = torch.as_tensor(
        y, dtype=torch.long if task == "classificacao" else torch.float32
    )
    x_tensor = torch.from_numpy(x)
    return DatasetBundle(
        name=name,
        task=task,
        kind="tabela",
        train=TensorDataset(x_tensor[train_idx], y_tensor[train_idx]),
        test=TensorDataset(x_tensor[test_idx], y_tensor[test_idx]),
        source=source,
        num_outputs=num_outputs,
        model_params={"in_features": x.shape[1]},
        description=description,
    )


def pad_sequences(sequences, max_len):
    """Corta/completa com zeros (no fim) cada sequência para ``max_len``."""
    result = np.full((len(sequences), max_len), PAD_INDEX, dtype=np.int64)
    for row, sequence in enumerate(sequences):
        sequence = list(sequence)[:max_len]
        result[row, :len(sequence)] = sequence
    return result


def text_bundle(name, source, description, train_x, train_y, test_x, test_y,
                num_classes, vocab_size):
    return DatasetBundle(
        name=name,
        task="classificacao",
        kind="texto",
        train=TensorDataset(torch.from_numpy(train_x),
                            torch.as_tensor(train_y, dtype=torch.long)),
        test=TensorDataset(torch.from_numpy(test_x),
                           torch.as_tensor(test_y, dtype=torch.long)),
        source=source,
        num_outputs=num_classes,
        model_params={"vocab_size": vocab_size},
        description=description,
    )


# ---------------------------------------------------------------------------
# Imagem
# ---------------------------------------------------------------------------

def read_mnist_csv(csv_path):
    """Lê um CSV no formato MNIST: rótulo + 784 pixels por linha, sem
    cabeçalho."""
    csv_path = Path(csv_path)
    labels = []
    images = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        for row_number, row in enumerate(csv.reader(file), start=1):
            if not row:
                continue
            if len(row) != CSV_COLUMN_COUNT:
                raise ValueError(
                    f"{csv_path}, linha {row_number}: esperado "
                    f"{CSV_COLUMN_COUNT} valores (rótulo e 784 pixels), "
                    f"encontrados {len(row)}."
                )
            try:
                values = np.asarray(row, dtype=np.int64)
            except (OverflowError, ValueError) as error:
                raise ValueError(
                    f"{csv_path}, linha {row_number}: todos os valores devem "
                    "ser inteiros e o arquivo não deve conter cabeçalho."
                ) from error
            label, pixels = int(values[0]), values[1:]
            if not 0 <= label < 10:
                raise ValueError(
                    f"{csv_path}, linha {row_number}: rótulo {label} fora do "
                    "intervalo 0–9."
                )
            if np.any((pixels < 0) | (pixels > 255)):
                raise ValueError(
                    f"{csv_path}, linha {row_number}: os pixels devem estar no "
                    "intervalo 0–255."
                )
            labels.append(label)
            images.append(pixels.astype(np.uint8))
    if not images:
        raise ValueError(f"O arquivo CSV está vazio: {csv_path}")
    return (np.asarray(images, dtype=np.uint8).reshape(-1, 1, 28, 28),
            np.asarray(labels, dtype=np.int64))


def load_mnist(data_dir, train_csv=None, test_csv=None):
    if train_csv and test_csv:
        train_x, train_y = read_mnist_csv(train_csv)
        test_x, test_y = read_mnist_csv(test_csv)
        source = f"CSV: {train_csv} | {test_csv}"
    else:
        path = download(KERAS_URL + "mnist.npz", data_dir / "mnist" / "mnist.npz")
        with np.load(path) as data:
            train_x = data["x_train"][:, None]
            train_y = data["y_train"]
            test_x = data["x_test"][:, None]
            test_y = data["y_test"]
        source = KERAS_URL + "mnist.npz"
    return image_bundle(
        "mnist", source, "Dígitos manuscritos 28x28 em tons de cinza",
        train_x, train_y, test_x, test_y, 10,
    )


def read_idx(path, offset):
    with gzip.open(path, "rb") as file:
        return np.frombuffer(file.read(), dtype=np.uint8, offset=offset)


def load_fashion_mnist(data_dir, **_):
    folder = data_dir / "fashion_mnist"
    files = {}
    for key, filename in {
        "train_y": "train-labels-idx1-ubyte.gz",
        "train_x": "train-images-idx3-ubyte.gz",
        "test_y": "t10k-labels-idx1-ubyte.gz",
        "test_x": "t10k-images-idx3-ubyte.gz",
    }.items():
        files[key] = download(KERAS_URL + filename, folder / filename)
    return image_bundle(
        "fashion_mnist", KERAS_URL + "(Fashion-MNIST)",
        "70 mil roupas 28x28 em tons de cinza, 10 classes",
        read_idx(files["train_x"], 16).reshape(-1, 1, 28, 28),
        read_idx(files["train_y"], 8).astype(np.int64),
        read_idx(files["test_x"], 16).reshape(-1, 1, 28, 28),
        read_idx(files["test_y"], 8).astype(np.int64),
        10,
    )


def load_cifar10(data_dir, **_):
    archive = download(CIFAR10_URL, data_dir / "cifar10" / "cifar-10-python.tar.gz")
    batches = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name.startswith("data_batch_") or name == "test_batch":
                # Formato oficial do CIFAR-10 (pickle), baixado da fonte oficial.
                batches[name] = pickle.load(
                    tar.extractfile(member), encoding="bytes"
                )

    def join(names):
        x = np.concatenate([batches[n][b"data"] for n in names])
        y = np.concatenate([batches[n][b"labels"] for n in names])
        return x.reshape(-1, 3, 32, 32), y.astype(np.int64)

    train_x, train_y = join([f"data_batch_{i}" for i in range(1, 6)])
    test_x, test_y = join(["test_batch"])
    return image_bundle(
        "cifar10", CIFAR10_URL, "60 mil fotos coloridas 32x32, 10 classes",
        train_x, train_y, test_x, test_y, 10,
    )


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------

def load_imdb(data_dir, **_):
    path = download(KERAS_URL + "imdb.npz", data_dir / "imdb" / "imdb.npz")
    # O imdb.npz do Keras guarda listas de índices de palavras (objetos).
    with np.load(path, allow_pickle=True) as data:
        splits = {key: data[key] for key in ("x_train", "y_train",
                                             "x_test", "y_test")}

    def encode(sequences):
        # Índice = posição da palavra no ranking de frequência (1 = mais
        # comum). Reserva 0 para preenchimento e 1 para palavras raras.
        encoded = []
        for sequence in sequences:
            values = np.asarray(sequence, dtype=np.int64) + 1
            values[values >= IMDB_VOCAB_SIZE] = OOV_INDEX
            encoded.append(values)
        return pad_sequences(encoded, IMDB_MAX_LEN)

    return text_bundle(
        "imdb", KERAS_URL + "imdb.npz",
        "50 mil críticas de filmes, sentimento positivo/negativo",
        encode(splits["x_train"]), splits["y_train"],
        encode(splits["x_test"]), splits["y_test"],
        2, IMDB_VOCAB_SIZE,
    )


# ---------------------------------------------------------------------------
# Tabela
# ---------------------------------------------------------------------------

def load_breast_cancer_data(data_dir, **_):
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    return tabular_bundle(
        "breast_cancer", "sklearn.datasets.load_breast_cancer",
        "569 pacientes, tumor maligno/benigno (80% treino, 20% teste)",
        data.data, data.target.astype(np.int64), "classificacao", 2,
        stratify=True,
    )


# ---------------------------------------------------------------------------
# Áudio
# ---------------------------------------------------------------------------

def mel_filterbank():
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    n_freqs = N_FFT // 2 + 1
    freqs = np.linspace(0, SAMPLE_RATE / 2, n_freqs)
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(SAMPLE_RATE / 2),
                             N_MELS + 2)
    hz_points = mel_to_hz(mel_points)
    bank = np.zeros((N_MELS, n_freqs), dtype=np.float32)
    for i in range(N_MELS):
        left, center, right = hz_points[i:i + 3]
        rising = (freqs - left) / (center - left)
        falling = (right - freqs) / (right - center)
        bank[i] = np.maximum(0, np.minimum(rising, falling))
    return torch.from_numpy(bank)


def log_mel_spectrograms(waveforms, bank):
    """(N, 16000) float -> (N, 1, N_MELS, frames)."""
    window = torch.hann_window(N_FFT)
    spec = torch.stft(torch.from_numpy(waveforms), N_FFT, HOP_LENGTH,
                      window=window, return_complex=True).abs().pow(2)
    mel = torch.matmul(bank, spec)
    return torch.log(mel + 1e-6).unsqueeze(1)


def read_wav(file_obj):
    with wave.open(file_obj, "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            return None
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    clip = np.zeros(SAMPLE_RATE, dtype=np.float32)
    audio = audio[:SAMPLE_RATE].astype(np.float32) / 32768.0
    clip[:len(audio)] = audio
    return clip


def prepare_speech_commands(data_dir):
    """Baixa o Speech Commands e gera o cache de espectrogramas (uma vez)."""
    folder = data_dir / "speech_commands"
    cache = folder / "speech_commands_logmel.npz"
    if cache.is_file():
        return cache
    archive = download(SPEECH_COMMANDS_URL,
                       folder / "speech_commands_v0.02.tar.gz")
    print("Processando o Speech Commands (só na primeira vez; pode levar "
          "alguns minutos)...")
    bank = mel_filterbank()
    word_index = {word: i for i, word in enumerate(SPEECH_WORDS)}
    test_list = set()
    paths, labels, features = [], [], []
    buffer, buffer_paths, buffer_labels = [], [], []

    def flush():
        if buffer:
            features.append(
                log_mel_spectrograms(np.stack(buffer), bank).numpy()
            )
            paths.extend(buffer_paths)
            labels.extend(buffer_labels)
            buffer.clear()
            buffer_paths.clear()
            buffer_labels.clear()

    # Leitura em streaming: nada é extraído para o disco.
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = member.name.removeprefix("./")
            if name == "testing_list.txt":
                content = tar.extractfile(member).read().decode()
                test_list = set(content.split())
                continue
            word = name.split("/")[0]
            if not (member.isfile() and name.endswith(".wav")
                    and word in word_index):
                continue
            # O tar em streaming não permite "seek"; lê o arquivo inteiro
            # (cerca de 32 KB) para a memória antes de abrir como WAV.
            clip = read_wav(io.BytesIO(tar.extractfile(member).read()))
            if clip is None:
                continue
            buffer.append(clip)
            buffer_paths.append(name)
            buffer_labels.append(word_index[word])
            if len(buffer) >= 1024:
                flush()
    flush()

    is_test = np.array([path in test_list for path in paths])
    # Grava num arquivo temporário e renomeia no fim: se o processamento
    # for interrompido, não fica um cache pela metade.
    partial = cache.with_name(cache.name + ".part")
    with partial.open("wb") as file:
        np.savez(file, x=np.concatenate(features),
                 y=np.asarray(labels, dtype=np.int64), is_test=is_test)
    partial.replace(cache)
    print(f"Cache salvo em: {cache}")
    return cache


def load_speech_commands(data_dir, **_):
    cache = prepare_speech_commands(data_dir)
    with np.load(cache) as data:
        x = torch.from_numpy(data["x"])
        y = torch.from_numpy(data["y"])
        is_test = torch.from_numpy(data["is_test"])
    train_x, test_x = x[~is_test], x[is_test]
    mean, std = train_x.mean(), train_x.std()
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    return DatasetBundle(
        name="speech_commands",
        task="classificacao",
        kind="imagem",   # espectrograma tratado como imagem de 1 canal
        train=TensorDataset(train_x, y[~is_test]),
        test=TensorDataset(test_x, y[is_test]),
        source=SPEECH_COMMANDS_URL,
        num_outputs=len(SPEECH_WORDS),
        model_params={"in_channels": 1},
        description="Comandos de voz de 1 s (10 palavras: "
        + ", ".join(SPEECH_WORDS) + "), como espectrograma log-mel",
    )


# ---------------------------------------------------------------------------
# Série temporal
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """Janelas de ``past`` passos para prever a temperatura ``future`` passos
    depois do fim da janela."""

    def __init__(self, features, target, starts, past, future):
        self.features = features
        self.target = target
        self.starts = starts
        self.past = past
        self.future = future

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        end = start + self.past
        return (self.features[start:end],
                self.target[end - 1 + self.future])


def load_jena_climate(data_dir, **_):
    archive = download(JENA_URL, data_dir / "jena_climate" /
                       "jena_climate_2009_2016.csv.zip")
    with zipfile.ZipFile(archive) as zip_file:
        csv_name = next(n for n in zip_file.namelist() if n.endswith(".csv"))
        with zip_file.open(csv_name) as raw:
            lines = raw.read().decode("utf-8").splitlines()
    reader = csv.reader(lines)
    header = next(reader)
    columns = header[1:]   # a primeira coluna é a data/hora
    values = np.asarray([row[1:] for row in reader if row], dtype=np.float32)
    values = values[::JENA_STEP]
    values[values == -9999.0] = 0.0   # leituras inválidas de vento

    temperature_index = columns.index(JENA_TEMPERATURE_COLUMN)
    temperature = torch.from_numpy(values[:, temperature_index].copy())

    split = int(len(values) * 0.7)
    mean = values[:split].mean(axis=0)
    std = values[:split].std(axis=0)
    std[std == 0] = 1.0
    features = torch.from_numpy((values - mean) / std)

    past, future = JENA_PAST_HOURS, JENA_FUTURE_HOURS
    last_start = len(values) - past - future
    # Treino: janelas cujo alvo fica nos primeiros 70%; teste: o restante,
    # sem sobreposição com o período de treino.
    train_starts = np.arange(0, split - past - future + 1)
    test_starts = np.arange(split, last_start + 1)
    return DatasetBundle(
        name="jena_climate",
        task="regressao",
        kind="serie_temporal",
        train=WindowDataset(features, temperature, train_starts, past, future),
        test=WindowDataset(features, temperature, test_starts, past, future),
        source=JENA_URL,
        num_outputs=1,
        model_params={"in_features": features.shape[1]},
        description=f"Clima de Jena (1 leitura/hora): {past} h de histórico "
        f"para prever a temperatura (°C) {future} h depois; 70% iniciais "
        "treino, 30% finais teste",
    )


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------

LOADERS = {
    "mnist": load_mnist,
    "fashion_mnist": load_fashion_mnist,
    "cifar10": load_cifar10,
    "imdb": load_imdb,
    "breast_cancer": load_breast_cancer_data,
    "speech_commands": load_speech_commands,
    "jena_climate": load_jena_climate,
}
DATASET_NAMES = list(LOADERS)


FASHION_FILES = ["train-labels-idx1-ubyte.gz", "train-images-idx3-ubyte.gz",
                 "t10k-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz"]

# Arquivos de cada dataset: (URL, caminho dentro de --data-dir). São os
# mesmos caminhos usados pelas funções load_*.
DOWNLOADS = {
    "mnist": [(KERAS_URL + "mnist.npz", "mnist/mnist.npz")],
    "fashion_mnist": [(KERAS_URL + name, f"fashion_mnist/{name}")
                      for name in FASHION_FILES],
    "cifar10": [(CIFAR10_URL, "cifar10/cifar-10-python.tar.gz")],
    "imdb": [(KERAS_URL + "imdb.npz", "imdb/imdb.npz")],
    "jena_climate": [(JENA_URL,
                      "jena_climate/jena_climate_2009_2016.csv.zip")],
}


def prepare_dataset(name, data_dir, use_csv=False):
    """Só baixa (e, no Speech Commands, pré-processa) o dataset, sem
    carregá-lo na memória. Permite deixar todos os downloads prontos antes de
    começar os treinos."""
    data_dir = Path(data_dir)
    if name == "mnist" and use_csv:
        return
    if name == "speech_commands":
        prepare_speech_commands(data_dir)
    elif name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer

        load_breast_cancer()   # já vem com o scikit-learn
    for url, relative_path in DOWNLOADS.get(name, []):
        download(url, data_dir / relative_path)


def load_dataset(name, data_dir, train_csv=None, test_csv=None):
    data_dir = Path(data_dir)
    if name == "mnist":
        return load_mnist(data_dir, train_csv=train_csv, test_csv=test_csv)
    return LOADERS[name](data_dir)
