#pragma once
#include <string>
#include <cstdint>

struct ModelConfig {
    std::string path         = "./models/modelq.gguf";
    int         n_ctx        = 8192;
    int         n_gpu_layers = -1;
    int         n_threads    = -1;
    int         n_batch      = 512;
    bool        flash_attn   = true;
    bool        mmap         = true;
    bool        mlock        = false;
};

struct InferenceConfig {
    float   temperature    = 0.7f;
    float   top_p          = 0.9f;
    int     top_k          = 40;
    float   repeat_penalty = 1.1f;
    int     repeat_last_n  = 64;
    int     max_tokens     = 4096;
    int32_t seed           = -1;
};

// "plan" — analysis only, no tools
// "build" — execute immediately, tools enabled
enum class AgentMode { Plan, Build };

struct AgentConfig {
    std::string stream_prompt = "You are a helpful AI assistant.";
    AgentMode   mode          = AgentMode::Build;
    bool        stream        = true;
    bool        verbose       = false;
    int         history_limit = 20;
    std::string sessions_dir  = "";   // "" = disabled; dir where named session .json files live

    // Per-mode system prompts (set in config.json)
    std::string plan_system_prompt =
        "Ты — аналитик и архитектор. Твоя задача: разобрать задачу и составить подробный план. "
        "НЕ создавай файлы и НЕ выполняй команды — только анализ, структура, пронумерованные шаги. "
        "Всегда отвечай на русском языке.";

    std::string build_system_prompt =
        "Ты — исполнитель. Выполняй задачи НЕМЕДЛЕННО без предисловий и объяснений. "
        "Используй инструменты сразу. Не описывай что собираешься делать — просто делай. "
        "Минимум текста, максимум действий. Всегда отвечай на русском языке.";
};

struct ToolsConfig {
    bool        enabled        = true;
    int         max_iterations = 10;
    std::string working_dir    = ".";
};

struct AppConfig {
    ModelConfig     model;
    InferenceConfig inference;
    AgentConfig     agent;
    ToolsConfig     tools;
};

AppConfig   load_config(const std::string& path);
std::string mode_name(AgentMode m);   // "plan" | "build"
AgentMode   mode_from_string(const std::string& s);
