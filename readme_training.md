# Como rodar o treinamento no seu computador

Este guia explica, passo a passo, como rodar o treinamento de redes neurais deste
projeto no seu computador. **Você não precisa saber programar.** Basta seguir os
passos na ordem e copiar e colar os comandos.

O programa vem pronto dentro de um "container" do Docker. Pense nele como uma
caixa fechada com tudo de que o treinamento precisa: o programa, as bibliotecas e
todos os dados. Você só instala o Docker, baixa essa caixa e manda rodar. No fim,
os resultados aparecem numa pasta do seu computador, numa planilha que abre no
Excel.

---

## Resumo rápido

1. Instale o **Docker Desktop** ([passo 1](#passo-1-instalar-o-docker-desktop)).
2. Crie uma pasta e abra o **PowerShell** nela ([passo 2](#passo-2-criar-uma-pasta-e-abrir-o-powershell)).
3. Baixe o programa ([passo 3](#passo-3-baixar-o-programa)):
   `docker pull luccachastinet/ml-training:1.0`
4. Rode um teste rápido ([passo 4](#passo-4-fazer-um-teste-rápido)).
5. Rode o treinamento completo ([passo 5](#passo-5-rodar-o-treinamento-completo)).
6. Abra os resultados no Excel ([passo 6](#passo-6-ver-os-resultados)).

---

## O que você precisa

- Computador com **Windows 10 ou 11** (64 bits). Para Mac ou Linux, veja o
  [final deste guia](#mac-e-linux).
- **Pelo menos 8 GB de memória RAM** (16 GB ou mais é o ideal).
- **Pelo menos 20 GB livres** no disco.
- **Internet**, para baixar o Docker e o programa (cerca de 7 GB no total).
- **Placa de vídeo NVIDIA (opcional).** Sem ela, o treinamento roda normalmente
  no processador. Com ela, também dá para testar com a placa de vídeo
  ([passo 5](#com-placa-de-vídeo-nvidia)).

> 💡 **Não sabe quanta memória ou quantos processadores o seu computador tem?**
> Aperte **Ctrl + Shift + Esc** para abrir o Gerenciador de Tarefas, clique em
> **Desempenho** e depois em **CPU**. Veja o número em
> **Processadores lógicos**: você vai precisar dele no passo 5. Clique em
> **Memória** para ver a RAM.

---

## Passo 1: Instalar o Docker Desktop

O Docker é o programa que abre e roda o "container". Ele é gratuito para uso
pessoal e educacional.

1. Acesse **https://www.docker.com/products/docker-desktop/** e clique em
   **Download for Windows**.
   - Se o site perguntar, escolha a versão **AMD64**. Ela serve para a grande
     maioria dos computadores (Intel ou AMD).
2. Abra o arquivo baixado (**Docker Desktop Installer.exe**).
3. Se o Windows perguntar se deseja permitir alterações, clique em **Sim**.
4. Na tela de configuração, **deixe marcada** a opção
   **Use WSL 2 instead of Hyper-V (recommended)** e clique em **OK**.
5. Espere a instalação terminar e clique em **Close and restart** (ou reinicie o
   computador).
6. Depois de reiniciar, o Docker Desktop abre sozinho. Se não abrir, procure por
   **Docker Desktop** no menu Iniciar.
7. **Aceite os termos de uso** (Accept).
8. Se o Docker pedir login ou criação de conta, você pode clicar em
   **Skip** (pular). Não é necessário ter conta.
9. Espere até aparecer **Engine running** (com um ícone verde) no canto inferior
   esquerdo da janela do Docker Desktop.

> ⚠️ **O Docker Desktop precisa estar aberto** (com "Engine running") sempre
> que você for rodar o treinamento.

Se aparecer algum erro aqui, veja [Problemas comuns](#problemas-comuns),
principalmente os itens sobre **WSL** e **virtualização**.

---

## Passo 2: Criar uma pasta e abrir o PowerShell

1. Crie uma pasta nova para o treinamento, por exemplo na Área de Trabalho, com
   o nome **treino**.
2. Abra essa pasta (dê dois cliques nela).
3. Clique na **barra de endereço** no topo da janela (onde aparece o caminho da
   pasta), apague o texto, digite **powershell** e aperte **Enter**.

   Uma janela azul ou preta vai abrir: esse é o **PowerShell**. Ele já abre
   "dentro" da pasta **treino**, e é lá que os resultados serão salvos.

> 💡 **Como colar comandos no PowerShell:** copie o comando deste guia (Ctrl+C),
> clique dentro da janela do PowerShell e clique com o **botão direito** do mouse
> (ou aperte Ctrl+V). Depois aperte **Enter**.

---

## Passo 3: Baixar o programa

No PowerShell, cole o comando abaixo e aperte **Enter**:

```powershell
docker pull luccachastinet/ml-training:1.0
```

O download tem cerca de **6,6 GB** e pode levar de alguns minutos a mais de uma
hora, conforme a sua internet. Você vai ver várias linhas de progresso. Quando
terminar, aparece uma linha parecida com:

```
Status: Downloaded newer image for luccachastinet/ml-training:1.0
```

Esse passo só precisa ser feito **uma vez**.

---

## Passo 4: Fazer um teste rápido

Antes do treinamento completo, faça um teste curto (menos de 1 minuto) para
confirmar que tudo funciona. Cole no PowerShell:

```powershell
docker run -it --rm --shm-size=4g -v "${PWD}\resultados:/app/results" luccachastinet/ml-training:1.0 --dataset breast_cancer --cpu-amount 1,2
```

Vão aparecer várias linhas de texto. No final, você deve ver algo assim:

```
Resumo:
  breast_cancer     1 CPU(s) data    sem GPU: 0.06s | acurácia de teste 97.35%
  breast_cancer     2 CPU(s) data    sem GPU: 0.06s | acurácia de teste ...
```

Depois do resumo, aparece uma linha terminando em **#**. Digite **exit** e
aperte **Enter** para sair.

Abra a pasta **treino**: deve existir uma pasta nova chamada **resultados**.
**Se ela existe, está tudo funcionando.** 🎉

---

## Passo 5: Rodar o treinamento completo

### Escolher a quantidade de CPUs

O treinamento é repetido com quantidades diferentes de processadores (CPUs) para
comparar a velocidade. Isso é definido em `--cpu-amount`. Use apenas valores
**até o número de processadores lógicos** do seu computador (veja a dica no
início deste guia).

| Processadores lógicos do seu computador | Use |
| --- | --- |
| 4 | `--cpu-amount 1,2,4` |
| 8 | `--cpu-amount 1,2,4,8` |
| 12 a 15 | `--cpu-amount 1,2,4,8` |
| 16 ou mais | `--cpu-amount 1,2,4,8,16` |

> ⚠️ Com **8 GB de RAM** ou menos, não use 16. Use no máximo `1,2,4,8`.

### Sem placa de vídeo NVIDIA

Cole o comando abaixo, trocando `1,2,4,8,16` pela sua escolha da tabela:

```powershell
docker run -it --rm --shm-size=4g -v "${PWD}\resultados:/app/results" luccachastinet/ml-training:1.0 --dataset all --cpu-amount 1,2,4,8,16 --parallelism both
```

### Com placa de vídeo NVIDIA

Se o seu computador tem placa NVIDIA, primeiro confirme que o driver está
instalado. No PowerShell, cole:

```powershell
nvidia-smi
```

Deve aparecer uma tabela com o nome da sua placa e, no canto superior direito,
**CUDA Version: 13.0** ou maior. Se der erro, ou se a versão for menor, instale
ou atualize o driver em **https://www.nvidia.com/pt-br/drivers/** e reinicie o
computador.

Depois, rode o treinamento com a placa de vídeo:

```powershell
docker run -it --rm --gpus all --shm-size=4g -v "${PWD}\resultados:/app/results" luccachastinet/ml-training:1.0 --dataset all --cpu-amount 1,2,4,8,16 --use-gpu true --parallelism both
```

### Quanto tempo demora

**Muito.** O treinamento completo faz dezenas de treinos seguidos (até 140 com
placa de vídeo), e o tempo depende do computador. Em geral leva **várias horas**.
É recomendado deixar rodando durante a noite.

Enquanto roda:

- **Não feche o PowerShell nem o Docker Desktop.**
- Deixe o computador ligado na tomada e desative a suspensão automática
  (Configurações → Sistema → Energia → Tela e suspensão → **Nunca**).
- Linhas como `Época [3/10]` mostram o progresso de cada treino.
- Pode ser normal o computador ficar lento e com o ventilador alto.

Quando terminar, aparece o **Resumo** com todos os treinos e, depois, uma linha
terminando em **#**. Digite **exit** e aperte **Enter**.

> 💡 **Precisa parar no meio?** Aperte **Ctrl + C** no PowerShell. Os treinos que
> já terminaram continuam salvos na pasta **resultados**.

---

## Passo 6: Ver os resultados

Abra a pasta **treino** → **resultados**. Dentro dela:

| Arquivo ou pasta | O que é |
| --- | --- |
| **all_results.csv** | **Planilha com todos os resultados.** Abra com o Excel (dois cliques). |
| all_results.json | Os mesmos resultados, num formato para programas. |
| Uma pasta por dataset (ex.: **cifar10**) | Para cada treino: os resultados e os gráficos (arquivos .png) da evolução do treino. |

Na planilha **all_results.csv**, cada linha é um treino. As colunas mais
importantes:

| Coluna | Significado |
| --- | --- |
| `dataset` | Conjunto de dados usado no treino (imagens, textos, áudio etc.) |
| `paralelismo` | `data`: o trabalho foi dividido entre vários processos (workers). `threads`: um único processo usando vários processadores. |
| `cpus` | Quantidade de processadores usados |
| `gpu` | `True` se usou a placa de vídeo |
| `tempo_total_segundos` | **Quanto tempo o treino levou** |
| `acc_final_teste` | Acurácia: porcentagem de acertos do modelo em dados que ele não viu no treino |
| `mae_final_teste` | Erro médio (só no dataset de clima, em °C) |
| `cpu_modelo`, `ram_total_gb` | Informações do computador onde o treino rodou |

**Para enviar os resultados**, mande a pasta **resultados** inteira. Para isso,
clique com o botão direito nela → **Compactar para arquivo ZIP**. Se preferir
mandar um arquivo só, mande o **all_results.csv**.

> ℹ️ Algumas colunas, como `sistema_operacional` (Debian) e `ram_velocidade_mhz`
> ("não disponível"), mostram informações de dentro do container, e não do seu
> Windows. Isso é normal.

---

## Problemas comuns

### "docker" não é reconhecido como comando

O Docker não está instalado ou o PowerShell foi aberto antes da instalação.
Feche o PowerShell, confirme que o Docker Desktop está instalado e aberto, e abra
o PowerShell de novo (passo 2).

### "error during connect" ou "Cannot connect to the Docker daemon"

O Docker Desktop não está aberto. Abra o **Docker Desktop** pelo menu Iniciar,
espere aparecer **Engine running** e rode o comando de novo.

### O Docker Desktop fala em "WSL" ou pede para atualizar o WSL

Abra o menu Iniciar, procure **PowerShell**, clique com o botão direito →
**Executar como administrador** e cole:

```powershell
wsl --install
```

Depois que terminar, **reinicie o computador** e abra o Docker Desktop de novo.

### "Virtualization support not detected" (virtualização desativada)

A virtualização precisa estar ativada na BIOS do computador. Isso é feito ao
ligar o computador, apertando uma tecla como **F2**, **F10**, **Del** ou **Esc**
(muda de fabricante para fabricante). Procure uma opção chamada
**Intel Virtualization Technology (VT-x)**, **SVM Mode** ou **AMD-V** e ative.
Se não se sentir seguro, peça ajuda ou pesquise "ativar virtualização" junto com
o modelo do seu computador.

### "precisa de pelo menos 2 GB de memória compartilhada"

Faltou `--shm-size=4g` no comando. Copie o comando inteiro deste guia de novo.

### "could not select device driver" ou erro com "--gpus"

O computador não tem placa NVIDIA, ou o driver não está instalado. Use o comando
**sem placa de vídeo** do passo 5.

### "nenhuma GPU NVIDIA (CUDA) está disponível"

O comando tem `--use-gpu true`, mas o Docker não encontrou a placa NVIDIA.
Confira o `nvidia-smi` (passo 5) ou use o comando **sem placa de vídeo**.

### O treinamento para com "Killed" ou o Docker fecha sozinho

Faltou memória RAM. Rode de novo com menos CPUs, por exemplo
`--cpu-amount 1,2,4`. Se quiser, também dá para dar mais memória ao Docker:
abra o Docker Desktop → **Settings** (ícone de engrenagem) → **Resources** (se
essa opção aparecer).

### A pasta "resultados" não apareceu

Confira se você abriu o PowerShell **dentro** da pasta treino (passo 2) e se o
comando tem a parte `-v "${PWD}\resultados:/app/results"`.

### Estou usando o Prompt de Comando (CMD), não o PowerShell

Os comandos deste guia são para o **PowerShell**. No CMD, troque `${PWD}` por
`%cd%`. Por exemplo:
`-v "%cd%\resultados:/app/results"`.

---

## Liberar espaço depois

Quando não precisar mais do programa, apague-o para liberar os ~7 GB. No
PowerShell:

```powershell
docker rmi luccachastinet/ml-training:1.0
```

Depois, se quiser, desinstale o **Docker Desktop** em Configurações → Aplicativos.
A pasta **resultados** não é afetada.

---

## Mac e Linux

### Linux

1. Instale o Docker seguindo **https://docs.docker.com/engine/install/**.
2. Para usar placa NVIDIA, instale também o **NVIDIA Container Toolkit**:
   **https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html**.
3. Use os mesmos comandos deste guia num terminal, trocando
   `"${PWD}\resultados:/app/results"` por `"$(pwd)/resultados:/app/results"`.
   Se aparecer "permission denied", coloque `sudo` antes de `docker`.

### Mac

Instale o Docker Desktop para Mac (**https://www.docker.com/products/docker-desktop/**)
e use os comandos deste guia no aplicativo **Terminal**, trocando
`"${PWD}\resultados:/app/results"` por `"$(pwd)/resultados:/app/results"`.

- **Não use** `--gpus all` nem `--use-gpu true`, porque Macs não têm placa NVIDIA.
- Em Macs com chip **Apple (M1, M2, M3, M4…)**, o programa roda em modo de
  compatibilidade e fica **bem mais lento**. Os tempos medidos não devem ser
  comparados com os de outros computadores.

---

## O que o programa faz (para quem tiver curiosidade)

O programa treina redes neurais em 7 conjuntos de dados públicos:

| Dataset | O que é |
| --- | --- |
| mnist | Fotos de dígitos escritos à mão (0 a 9) |
| fashion_mnist | Fotos de roupas (camiseta, calça, tênis…) |
| cifar10 | Fotos coloridas de aviões, carros, animais… |
| imdb | Críticas de filmes (positiva ou negativa) |
| breast_cancer | Exames de tumores (maligno ou benigno) |
| speech_commands | Gravações de palavras faladas ("yes", "no", "stop"…) |
| jena_climate | Medições do clima, para prever a temperatura |

Cada treino é repetido com quantidades diferentes de processadores, de duas
formas (`--parallelism both`):

- **Paralelismo de dados (`data`):** os dados são divididos entre vários
  processos (workers), que treinam ao mesmo tempo e combinam o que aprenderam a
  cada passo.
- **Threads (`threads`):** um único processo usa vários processadores ao mesmo
  tempo em cada cálculo.

Comparando os tempos, dá para ver quanto o treinamento acelera com mais
processadores em cada forma de paralelismo e, se houver, com a placa de vídeo.
