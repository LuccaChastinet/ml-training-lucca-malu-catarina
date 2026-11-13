# Treino de redes neurais em vários datasets

Este projeto treina redes neurais com PyTorch em vários datasets públicos
(imagem, texto, tabela, áudio e série temporal) e compara o tempo de treino com
diferentes quantidades de CPUs e, opcionalmente, com uma GPU NVIDIA (CUDA).

O foco é o **paralelismo de dados**: o treino é dividido entre vários workers
(processos), cada um com uma cópia do modelo e uma parte diferente dos dados.
Para comparação, também é possível treinar em um único processo com várias
threads (veja [Modos de paralelismo](#modos-de-paralelismo)).

## Datasets disponíveis

Escolha o dataset pelo nome com `--dataset`. Os datasets são baixados
automaticamente na primeira vez e ficam guardados na pasta `data/` (veja
`--data-dir`), então as próximas execuções não baixam de novo.

| Nome (`--dataset`) | Tipo | Conteúdo | Tarefa | Modelo | Download |
| --- | --- | --- | --- | --- | --- |
| `mnist` | Imagem | 70 mil dígitos manuscritos 28x28 em tons de cinza | Classificação (10 classes) | CNN | 11 MB |
| `fashion_mnist` | Imagem | 70 mil roupas 28x28 em tons de cinza | Classificação (10 classes) | CNN | 30 MB |
| `cifar10` | Imagem | 60 mil fotos coloridas 32x32 | Classificação (10 classes) | CNN | 163 MB |
| `imdb` | Texto | 50 mil críticas de filmes | Sentimento positivo/negativo | CNN de texto | 17 MB |
| `breast_cancer` | Tabela | 569 pacientes, 30 medidas do tumor | Maligno/benigno | MLP | já vem no scikit-learn |
| `speech_commands` | Áudio | Comandos de voz de 1 s: yes, no, up, down, left, right, on, off, stop, go | Classificação (10 palavras) | CNN sobre espectrograma | **2,4 GB** |
| `jena_climate` | Série temporal | Clima de Jena (2009–2016), leitura a cada 10 min | Prever a temperatura 24 h depois (regressão) | LSTM | 13 MB |

Formas de escolher:

- Um dataset: `--dataset cifar10`
- Vários, separados por vírgula (sem espaços), na ordem informada:
  `--dataset mnist,imdb,jena_climate`
- Todos, um após o outro, na ordem da tabela: `--dataset all`

Os dados vêm das mesmas fontes oficiais usadas por `tensorflow.keras.datasets`,
`tensorflow_datasets` e `sklearn.datasets`, mas sem precisar instalar o
TensorFlow.

Detalhes de cada dataset:

- **Treino e teste:** `mnist`, `fashion_mnist`, `cifar10`, `imdb` e
  `speech_commands` usam a divisão oficial. O `breast_cancer` não tem divisão
  oficial: é separado em 80% treino e 20% teste, sempre da mesma forma (seed
  fixa). No `jena_climate`, os 70%
  iniciais do período são treino e os 30% finais são teste.
- **Texto:** o `imdb` usa as 10 mil palavras mais frequentes e os 256
  primeiros termos de cada crítica.
- **Áudio:** o `speech_commands` é convertido em espectrograma log-mel (40 x 51).
  Na primeira execução, o processamento do arquivo de 2,4 GB leva alguns minutos
  e gera um cache de cerca de 300 MB. Depois disso, carrega em segundos.
- **Série temporal:** o `jena_climate` usa uma leitura por hora e as últimas
  120 horas (5 dias) para prever a temperatura 24 horas depois.

Classificação é medida pela **acurácia** (%). O `jena_climate` é regressão e é
medido pelo **MAE** (erro absoluto médio), em °C.

## Execução local

É necessário ter uma versão do Python compatível com PyTorch. No Windows, abra
o CMD na pasta do projeto e instale as dependências:

```cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para usar GPU (somente NVIDIA), instale a versão do PyTorch com CUDA compatível
com o driver, por exemplo `pip install torch --index-url https://download.pytorch.org/whl/cu124`.

Exemplos:

```cmd
python training.py --dataset fashion_mnist --cpu-amount 4
python training.py --dataset mnist,cifar10,imdb --cpu-amount 1,2,4
python training.py --dataset all --cpu-amount 1,2,4,8,16
python training.py --dataset all --cpu-amount 1,2,4 --use-gpu true
python training.py --dataset all --cpu-amount 1,2,4,8,16 --parallelism both
```

### Execução completa (todos os datasets, até 16 CPUs, com GPU, threads e paralelismo de dados)

Roda todos os datasets com 1, 2, 4, 8 e 16 CPUs, com e sem GPU NVIDIA, nos dois
modos de paralelismo:

```cmd
python training.py --dataset all --cpu-amount 1,2,4,8,16 --use-gpu true --parallelism both
```

No Docker:

```cmd
docker run -it --gpus all --shm-size=4g --name treino-completo ml-training --dataset all --cpu-amount 1,2,4,8,16 --use-gpu true --parallelism both
```

São **140 treinos**: 7 datasets × 5 quantidades de CPUs × 2 (com e sem GPU) ×
2 modos (`data` e `threads`), de 10 épocas cada. Isso pode levar muitas horas. Os
resultados vão sendo salvos em `results/all_results.csv` ao fim de cada treino.
O computador precisa de GPU NVIDIA (veja [GPU no Docker](#gpu-no-docker)). Sem
NVIDIA, remova `--use-gpu true` e, no Docker, `--gpus all`. Nesse caso são 70
treinos.

### Argumentos

| Argumento | Obrigatório | Descrição |
| --- | --- | --- |
| `--dataset` | Sim | Nome do dataset, lista separada por vírgula ou `all` |
| `--cpu-amount` | Sim, exceto com `--download-only` | Quantidades de CPUs, separadas por vírgula (ex.: `4` ou `1,2,4`) |
| `--parallelism` | Não | `data` (padrão), `threads` ou `both` |
| `--use-gpu` | Não | `true` ou `false` (padrão: `false`) |
| `--data-dir` | Não | Pasta onde os datasets ficam guardados (padrão: `data`) |
| `--download-only` | Não | Só baixa e prepara os datasets, sem treinar (aí `--cpu-amount` não é necessário) |
| CSV de treino e `--test-csv` | Não | Só para o `mnist`: usa CSVs próprios em vez de baixar |

`--cpu-amount` define quantas CPUs cada treino usa. Com o paralelismo de dados
(padrão), cada CPU é um worker. Cada valor gera um treino, executado na ordem
informada:

- `4`: um único treino com 4 CPUs.
- `1,2,4`: três treinos, com 1, 2 e 4 CPUs. Funciona com qualquer quantidade de
  valores (ex.: `2,3,4,5,6`).

Valores acima do número de CPUs visíveis no computador/Docker geram um aviso e
não ganham desempenho.

### Modos de paralelismo

`--parallelism` escolhe como as N CPUs de cada valor de `--cpu-amount` são usadas:

| Modo | Processos | Threads por processo | Como funciona |
| --- | --- | --- | --- |
| `data` (padrão) | N workers | 1 | **Paralelismo de dados** com `DistributedDataParallel` (backend `gloo`). Cada worker tem uma cópia do modelo e treina numa parte diferente dos dados de treino (`DistributedSampler`). Depois de cada batch, os gradientes de todos os workers são somados (all-reduce), e todas as cópias do modelo recebem a mesma atualização. |
| `threads` | 1 | N | Um único processo, com cada operação (convolução, multiplicação de matrizes…) dividida entre N threads (`torch.set_num_threads`). Serve de base de comparação. |
| `both` | | | Roda `data` e depois `threads` para cada quantidade de CPUs. |

Com 1 CPU, os dois modos são equivalentes: 1 processo com 1 thread.

No paralelismo de dados, cada worker processa **64 exemplos por batch**. Assim,
o batch global (o total de exemplos por atualização do modelo) é 64 × N: com 4
workers, cada atualização usa 256 exemplos. Isso reduz o número de atualizações
por época. Em datasets pequenos, como o `breast_cancer`, a acurácia final pode
cair com muitos workers. As colunas `batch_por_worker` e `batch_global` registram
esses valores.

A avaliação no conjunto de teste também é dividida entre os workers, e os
resultados são somados. O tempo medido começa quando todos os workers estão
prontos, então não inclui a criação dos processos.

`--use-gpu`:

- `false`: cada quantidade de CPUs gera um treino somente na CPU.
- `true`: cada quantidade de CPUs gera **dois** treinos, um com GPU NVIDIA e
  outro sem GPU. Por exemplo, `--cpu-amount 1,2 --use-gpu true` executa, nesta
  ordem: 1 CPU com GPU, 1 CPU sem GPU, 2 CPUs com GPU e 2 CPUs sem GPU.

Somente GPUs NVIDIA (CUDA) são aceitas. Se `--use-gpu true` for informado e não
houver GPU NVIDIA disponível para o PyTorch (incluindo GPUs AMD), o script
encerra com erro antes de treinar.

Com `--use-gpu true` e paralelismo de dados, todos os workers usam a mesma GPU.

A ordem completa é: para cada dataset, para cada quantidade de CPUs, com GPU (se
ativada) e depois sem GPU, e para cada modo de paralelismo. Exemplo:
`--dataset all --cpu-amount 1,2,4,8,16 --use-gpu true --parallelism both` faz
7 datasets × 5 quantidades × 2 (GPU) × 2 (modos) = 140 treinos de 10 épocas.
Isso pode levar muitas horas. Para testar, comece com um dataset pequeno e
poucas CPUs.

A execução tem duas fases. Primeiro, **todos** os datasets escolhidos são
baixados e preparados. Só depois começam os treinos, um após o outro, sem paradas
para download. Cada download tem timeout e até 5 tentativas. Se for
interrompido, o próximo download continua de onde parou.

Com vários datasets, se um deles não puder ser baixado (por exemplo, por falta de
internet), o script avisa, pula esse dataset e continua com os outros.

### MNIST a partir de CSVs próprios

O `mnist` pode usar os CSVs deste projeto em vez de baixar. Informe o CSV de
treino e `--test-csv`:

```cmd
python training.py mnist_train_small.csv --test-csv mnist_test.csv --dataset mnist --cpu-amount 4
```

Cada linha do CSV deve conter 785 valores inteiros, sem cabeçalho: primeiro o
rótulo do dígito (`0` a `9`) e depois os 784 pixels da imagem 28 × 28 (`0` a
`255`). Com `--dataset all`, os CSVs informados são usados no `mnist` e os outros
datasets são baixados normalmente.

## Resultados

Todos os treinos rodam por 10 épocas, com lotes de 64 exemplos por worker e
otimizador Adam. Cada treino é salvo em uma pasta própria:

- `results/<dataset>/workers_N/`: paralelismo de dados com N workers;
- `results/<dataset>/threads_N/`: 1 processo com N threads;
- com GPU, o nome da pasta termina em `_gpu` (ex.: `results/cifar10/workers_4_gpu/`).

Em cada pasta ficam:

- `results.json`;
- `training_loss.png`;
- `test_accuracy.png` (classificação) ou `test_mae.png` (regressão).

Todos os treinos da execução são reunidos em:

- `results/all_results.csv`: uma linha por treino, com separador `;` e decimais
  com vírgula, para abrir direto no Excel;
- `results/all_results.json`: o mesmo conteúdo em JSON.

Esses arquivos são atualizados ao fim de cada treino. Se a execução for
interrompida, os treinos já concluídos não se perdem.

Principais colunas:

| Coluna | Conteúdo |
| --- | --- |
| `dataset`, `tipo_dado`, `tarefa` | Dataset, tipo de dado e se é classificação ou regressão |
| `modo`, `paralelismo` | `data_parallel_ddp` (`data`) ou `processo_unico_threads` (`threads`) |
| `cpus`, `workers`, `threads_por_worker` | CPUs do treino e como foram divididas |
| `batch_por_worker`, `batch_global`, `backend_distribuido` | Tamanho do batch e backend de comunicação (`gloo`) |
| `gpu`, `gpu_nome` | Se a GPU foi usada |
| `tempo_total_segundos` | Tempo total do treino (10 épocas, incluindo a avaliação) |
| `acc_final_treino`, `acc_final_teste` | Acurácia (%), só em classificação |
| `mae_final_treino`, `mae_final_teste` | Erro absoluto médio, só em regressão |
| `loss_final_treino`, `loss_final_teste` | Loss final |
| `amostras_treino`, `amostras_teste`, `parametros_modelo` | Tamanho dos dados e do modelo |
| `sistema_operacional` | Nome e versão do sistema |
| `em_docker` | Se o treino rodou dentro de um container |
| `python_versao`, `pytorch_versao` | Versões do Python e do PyTorch |
| `cpu_modelo`, `cpu_nucleos_logicos` | Modelo do processador e núcleos lógicos |
| `gpu_modelo_nvidia` | Modelo da GPU NVIDIA, se houver |
| `ram_total_gb`, `ram_velocidade_mhz`, `ram_tipo` | Memória RAM total, velocidade e tipo (DDR4, DDR5…) |

Quando uma informação do sistema não pode ser lida, aparece `não disponível`.
Dentro do Docker, o sistema operacional é o Linux do container (não o Windows do
host), a RAM total é a disponível para o Docker/WSL e a velocidade e o tipo da
RAM normalmente não estão disponíveis, pois o container não tem acesso a essas
informações do hardware.

## Execução com Docker

A imagem já leva **todos os datasets prontos** dentro dela, na pasta
`/app/data`. Quem recebe a imagem não precisa baixar nenhum dataset: é só rodar.

### 1. Baixar os datasets (só quem constrói a imagem)

Na pasta do projeto, baixe e prepare todos os datasets na pasta `data/`, sem
treinar:

```cmd
python training.py --dataset all --download-only
```

Isso baixa cerca de 2,7 GB, incluindo os 2,4 GB do Speech Commands, e processa o
Speech Commands. Se for interrompido, rode o mesmo comando de novo: o que já foi
baixado é mantido e o restante continua de onde parou.

### 2. Construir a imagem

```cmd
docker build -t ml-training .
```

O `Dockerfile` copia a pasta `data/` para dentro da imagem. O arquivo original do
Speech Commands (2,4 GB) **não** entra, porque o cache processado (~300 MB) já
basta. Downloads incompletos também ficam de fora (veja o `.dockerignore`). Os
dados ocupam cerca de 550 MB na imagem. Se a pasta `data/` não existir, o
`docker build` falha: rode o passo 1 antes.

### 3. Distribuir a imagem

Para outra pessoa usar a imagem, há duas opções:

- Publicar em um registro, como o Docker Hub:

  ```cmd
  docker tag ml-training SEU_USUARIO/ml-training
  docker push SEU_USUARIO/ml-training
  ```

  A outra pessoa roda `docker pull SEU_USUARIO/ml-training` e usa esse nome no
  lugar de `ml-training`.

- Exportar para um arquivo e enviar o arquivo:

  ```cmd
  docker save -o ml-training.tar ml-training
  ```

  A outra pessoa carrega a imagem com `docker load -i ml-training.tar`.

### 4. Rodar

CMD ou PowerShell:

```cmd
docker run -it --shm-size=4g --name treino-1 ml-training --dataset all --cpu-amount 1,2,4,8,16
```

**Use sempre `--shm-size=4g`.** No paralelismo de dados, os workers compartilham
o dataset pela memória compartilhada (`/dev/shm`), que no Docker tem só 64 MB por
padrão. Sem essa opção, o script para logo no início com uma mensagem de erro.

O container inicia o treinamento e, ao terminar, abre um shell interativo. Os
resultados ficam dentro do container, em `/app/results`. No shell, liste ou
consulte os arquivos:

```sh
ls -R /app/results
cat /app/results/all_results.csv
```

O terminal permanece conectado ao shell até você digitar `exit`. O container é
mantido (não use `--rm`), então os resultados permanecem no seu filesystem
interno. Cada nova execução precisa de outro nome em `--name` (por exemplo,
`treino-2`).

### 5. Pegar os resultados

Há duas formas de trazer os resultados do container para o seu computador.

**Opção 1 (recomendada): salvar direto numa pasta do computador.** Adicione
`-v` ligando uma pasta do computador a `/app/results`. Os arquivos aparecem na
pasta à medida que cada treino termina, e continuam lá mesmo que o container seja
apagado. Execute na pasta onde quer os resultados; a subpasta `resultados` é
criada automaticamente.

CMD:

```cmd
docker run -it --shm-size=4g --name treino-1 -v "%cd%\resultados:/app/results" ml-training --dataset all --cpu-amount 1,2,4,8,16
```

PowerShell:

```powershell
docker run -it --shm-size=4g --name treino-1 -v "${PWD}\resultados:/app/results" ml-training --dataset all --cpu-amount 1,2,4,8,16
```

Linux/macOS:

```sh
docker run -it --shm-size=4g --name treino-1 -v "$(pwd)/resultados:/app/results" ml-training --dataset all --cpu-amount 1,2,4,8,16
```

A tabela final fica em `resultados\all_results.csv`. No Linux, os arquivos são
criados com o usuário `root`. Para criá-los com o seu usuário, adicione
`--user "$(id -u):$(id -g)"`.

**Opção 2: copiar depois do treino.** Se o container foi executado sem `-v`, os
resultados ficam dentro dele (o container não é apagado, porque não se usa
`--rm`). Em outro terminal, ou depois de sair com `exit`, copie a pasta com o
nome dado em `--name`:

```cmd
docker cp treino-1:/app/results ./resultados
```

O `docker cp` funciona mesmo com o container parado. Depois de copiar, o
container pode ser apagado com `docker rm treino-1`.

Para usar os CSVs do MNIST deste projeto, monte cada arquivo com `-v` (somente
leitura) e passe o caminho **dentro do container**. Os comandos abaixo devem ser
executados na pasta dos CSVs.

CMD:

```cmd
docker run -it --shm-size=4g --name treino-mnist -v "%cd%\mnist_train_small.csv:/data/train.csv:ro" -v "%cd%\mnist_test.csv:/data/test.csv:ro" ml-training /data/train.csv --test-csv /data/test.csv --dataset mnist --cpu-amount 4
```

PowerShell:

```powershell
docker run -it --shm-size=4g --name treino-mnist -v "${PWD}\mnist_train_small.csv:/data/train.csv:ro" -v "${PWD}\mnist_test.csv:/data/test.csv:ro" ml-training /data/train.csv --test-csv /data/test.csv --dataset mnist --cpu-amount 4
```

### GPU no Docker

Para usar `--use-gpu true` no Docker, o host precisa de uma GPU NVIDIA, do driver
NVIDIA e do NVIDIA Container Toolkit (no Windows, Docker Desktop com WSL 2). Passe
`--gpus all` no `docker run`:

```cmd
docker run -it --gpus all --shm-size=4g --name treino-gpu ml-training --dataset all --cpu-amount 1,2,4 --use-gpu true
```

No Linux, o `pip install torch` da imagem já instala a versão com CUDA.
