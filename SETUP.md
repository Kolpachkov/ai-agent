# Подготовка системы и сборка ai-agent

Руководство для чистой установки на Arch Linux / CachyOS (x86-64, NVIDIA GPU).  
Для Ubuntu/Debian команды пакетного менеджера отличаются — соответствующие варианты указаны.

---

## Содержание

1. [Требования](#1-требования)
2. [Системные пакеты](#2-системные-пакеты)
3. [Драйвер NVIDIA и CUDA](#3-драйвер-nvidia-и-cuda)
4. [Сборка llama.cpp из исходников](#4-сборка-llamacpp-из-исходников)
5. [nlohmann/json](#5-nlohmannjson)
6. [Клонирование и сборка проекта](#6-клонирование-и-сборка-проекта)
7. [Загрузка модели](#7-загрузка-модели)
8. [Запуск](#8-запуск)
9. [Конфигурация](#9-конфигурация)
10. [Устранение проблем](#10-устранение-проблем)

---

## 1. Требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| GPU | NVIDIA ≥ Pascal (compute 6.0) | RTX 2050+ (4 GB VRAM) |
| VRAM | 3 GB | 4 GB |
| RAM | 4 GB | 8 GB |
| Диск | 5 GB | 10 GB |
| ОС | Linux x86-64 | Arch / Ubuntu 22.04+ |
| GCC | ≥ 12 | ≥ 13 |
| CMake | ≥ 3.16 | ≥ 3.20 |
| CUDA | ≥ 11.8 | ≥ 12.0 |

---

## 2. Системные пакеты

### Arch Linux / CachyOS

```bash
sudo pacman -S --needed \
    base-devel git cmake ninja \
    readline \
    cuda cuda-tools
```

### Ubuntu 22.04 / Debian 12

```bash
sudo apt update && sudo apt install -y \
    build-essential git cmake ninja-build \
    libreadline-dev \
    pkg-config
```

> CUDA на Ubuntu устанавливается отдельно — см. раздел 3.

---

## 3. Драйвер NVIDIA и CUDA

### Arch Linux / CachyOS

```bash
# Драйвер + CUDA устанавливаются одним пакетом
sudo pacman -S --needed nvidia-dkms cuda

# Перезагрузка обязательна
sudo reboot
```

После перезагрузки проверить:

```bash
nvidia-smi          # должен показать GPU и версию драйвера
nvcc --version      # должен показать версию CUDA
```

### Ubuntu 22.04+

```bash
# Добавить репозиторий CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4 nvidia-driver-550

sudo reboot
```

---

## 4. Сборка llama.cpp из исходников

llama.cpp устанавливается глобально (`/usr/local`) — это нужно чтобы CMake нашёл его через `find_package(llama)`.

```bash
# Клонировать актуальную версию
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Создать build-директорию
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -G Ninja

# Собрать (используем все ядра)
cmake --build build -j$(nproc)

# Установить в /usr/local
sudo cmake --install build
```

Проверка:

```bash
ls /usr/local/include/llama.h      # заголовок
ls /usr/local/lib/cmake/llama/     # CMake-конфиг
ls /usr/local/lib/libllama.so      # библиотека
ls /usr/local/lib/libggml-cuda.so  # CUDA-бэкенд (нужен для GPU)
```

> Если `libggml-cuda.so` отсутствует — CUDA не была найдена при сборке.  
> Убедитесь что `nvcc` доступен в `PATH` и повторите cmake-шаг.

---

## 5. nlohmann/json

### Arch Linux / CachyOS

```bash
sudo pacman -S --needed nlohmann-json
```

### Ubuntu / Debian

```bash
sudo apt install -y nlohmann-json3-dev
```

### Из исходников (любой дистрибутив)

```bash
git clone --depth=1 https://github.com/nlohmann/json.git
cd json
cmake -B build -DJSON_BuildTests=OFF
sudo cmake --install build
```

---

## 6. Клонирование и сборка проекта

```bash
git clone <url-репозитория> ai-agent
cd ai-agent

# Настройка (Release с нативными оптимизациями)
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja

# Сборка
cmake --build build -j$(nproc)
```

Бинарник будет в `build/ai-agent`.

> **Debug-сборка:**
> ```bash
> cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
> cmake --build build-debug -j$(nproc)
> ```

---

## 7. Загрузка модели

Проект использует **Huihui-gemma-4-E2B** в формате GGUF.  
Скрипт загружает квантизованную версию Q4_K (~3.4 GB):

```bash
bash scripts/convert_model.sh
```

Файл сохранится в `models/model.gguf`.

### Альтернативные квантизации

| Файл | Размер | VRAM | Качество |
|------|--------|------|----------|
| `modelq3.gguf` (Q3_K_M) | ~2.1 GB | ~3 GB | базовое |
| `model.gguf` (Q4_K) | ~3.4 GB | ~4 GB | хорошее |

Для Q3_K_M измените путь в `config.json`:

```json
"model": {
    "path": "./models/modelq3.gguf"
}
```

> Модель можно скачать вручную с  
> `https://huggingface.co/huihui-ai/` — ищите файлы `*.Q3_K_M.gguf` или `*.Q4_K.gguf`.

---

## 8. Запуск

```bash
cd build          # или куда скопировали бинарник
cp ../config.json .
./ai-agent
```

При запросе рабочей папки укажите директорию, в которой агент будет читать/писать файлы (например `/home/user/Projects`).

### Флаги командной строки

```
--config <path>   путь к config.json (по умолчанию ./config.json)
--mode plan       запустить в режиме только-чтение
--mode build      запустить в режиме полного доступа (по умолчанию)
--verbose         показывать отладочный вывод llama.cpp
--no-stream       отключить потоковый вывод
--help            справка
```

---

## 9. Конфигурация

Основные параметры `config.json`:

```json
{
  "model": {
    "path": "./models/model.gguf",
    "n_ctx": 8192,          // размер контекста (токены)
    "n_gpu_layers": -1,     // -1 = все слои на GPU
    "flash_attn": true      // Flash Attention (рекомендуется)
  },
  "inference": {
    "temperature": 0.3,     // ниже = точнее, выше = креативнее
    "max_tokens": 8192
  },
  "agent": {
    "mode": "build",        // "plan" | "build"
    "history_limit": 20,    // макс. сообщений в контексте
    "history_file": "~/.cache/ai-agent/history.json"  // "" = отключить
  },
  "tools": {
    "max_iterations": 20,   // макс. tool calls за один запрос
    "working_dir": "/home/user/Projects"
  }
}
```

### Рекомендации по `n_gpu_layers`

| VRAM | `n_gpu_layers` |
|------|---------------|
| 3–4 GB | `-1` (Gemma-4 2B влезает целиком) |
| 2 GB | `20` (частичная выгрузка) |
| Нет GPU | `0` (только CPU, медленно) |

---

## 10. Устранение проблем

### `find_package(llama REQUIRED)` — не найдено

```bash
# Проверить установку
ls /usr/local/lib/cmake/llama/

# Если пусто — повторить установку llama.cpp (раздел 4)
# Если установлено в нестандартный путь:
cmake -B build -Dllama_DIR=/path/to/cmake/llama
```

### `libggml-cuda.so: cannot open shared object`

```bash
# Обновить кэш динамических библиотек
sudo ldconfig

# Проверить
ldconfig -p | grep ggml-cuda
```

### Агент не использует GPU (`n_gpu_layers = 0` в выводе)

```bash
# Проверить наличие CUDA-бэкенда
ls /usr/local/lib/libggml-cuda.so

# Пересобрать llama.cpp с явным указанием CUDA
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=$(which nvcc) ...
```

### Переполнение контекста (`Context overflow`)

Уменьшить `history_limit` или `max_iterations` в `config.json`,  
либо уменьшить `n_ctx` (при нехватке VRAM):

```json
"n_ctx": 4096,
"history_limit": 10
```

### `llama_decode failed` при генерации

Увеличить `n_ctx` или уменьшить `max_tokens` — KV-кэш заполнился:

```json
"max_tokens": 2048
```
