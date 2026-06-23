#include "config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

std::string mode_name(AgentMode m) {
    return m == AgentMode::Plan ? "plan" : "build";
}

AgentMode mode_from_string(const std::string& s) {
    return s == "plan" ? AgentMode::Plan : AgentMode::Build;
}

static void check_range(const char* name, int val, int lo, int hi) {
    if (val < lo || val > hi)
        throw std::runtime_error(std::string(name) + " must be in ["
            + std::to_string(lo) + ", " + std::to_string(hi) + "], got "
            + std::to_string(val));
}
static void check_range_f(const char* name, float val, float lo, float hi) {
    if (val < lo || val > hi)
        throw std::runtime_error(std::string(name) + " must be in ["
            + std::to_string(lo) + ", " + std::to_string(hi) + "], got "
            + std::to_string(val));
}

static void parse_model(const json& j, ModelConfig& m) {
    if (j.contains("path"))         m.path          = j["path"].get<std::string>();
    if (j.contains("n_ctx"))      { m.n_ctx         = j["n_ctx"].get<int>();
                                    check_range("n_ctx", m.n_ctx, 512, 131072); }
    if (j.contains("n_gpu_layers")) m.n_gpu_layers  = j["n_gpu_layers"].get<int>();
    if (j.contains("n_threads"))    m.n_threads     = j["n_threads"].get<int>();
    if (j.contains("n_batch"))    { m.n_batch       = j["n_batch"].get<int>();
                                    check_range("n_batch", m.n_batch, 1, 65536); }
    if (j.contains("flash_attn"))   m.flash_attn    = j["flash_attn"].get<bool>();
    if (j.contains("mmap"))         m.mmap          = j["mmap"].get<bool>();
    if (j.contains("mlock"))        m.mlock         = j["mlock"].get<bool>();
}

static void parse_inference(const json& j, InferenceConfig& i) {
    if (j.contains("temperature"))  { i.temperature    = j["temperature"].get<float>();
                                      check_range_f("temperature", i.temperature, 0.0f, 5.0f); }
    if (j.contains("top_p"))        { i.top_p          = j["top_p"].get<float>();
                                      check_range_f("top_p", i.top_p, 0.0f, 1.0f); }
    if (j.contains("top_k"))          i.top_k          = j["top_k"].get<int>();
    if (j.contains("repeat_penalty")) i.repeat_penalty = j["repeat_penalty"].get<float>();
    if (j.contains("repeat_last_n"))  i.repeat_last_n  = j["repeat_last_n"].get<int>();
    if (j.contains("max_tokens"))   { i.max_tokens     = j["max_tokens"].get<int>();
                                      check_range("max_tokens", i.max_tokens, 1, 131072); }
    if (j.contains("seed"))           i.seed           = j["seed"].get<int32_t>();
}

static void parse_agent(const json& j, AgentConfig& a) {
    if (j.contains("mode"))                a.mode                = mode_from_string(j["mode"].get<std::string>());
    if (j.contains("stream"))              a.stream              = j["stream"].get<bool>();
    if (j.contains("verbose"))             a.verbose             = j["verbose"].get<bool>();
    if (j.contains("history_limit"))       a.history_limit       = j["history_limit"].get<int>();
    if (j.contains("sessions_dir"))         a.sessions_dir        = j["sessions_dir"].get<std::string>();
    if (j.contains("plan_system_prompt"))  a.plan_system_prompt  = j["plan_system_prompt"].get<std::string>();
    if (j.contains("build_system_prompt")) a.build_system_prompt = j["build_system_prompt"].get<std::string>();
}

AppConfig load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open config file: " + path);

    json j;
    try {
        j = json::parse(f);
    } catch (const json::parse_error& e) {
        throw std::runtime_error(std::string("Config JSON parse error: ") + e.what());
    }

    AppConfig cfg;
    if (j.contains("model"))     parse_model(j["model"],         cfg.model);
    if (j.contains("inference")) parse_inference(j["inference"], cfg.inference);
    if (j.contains("agent"))     parse_agent(j["agent"],         cfg.agent);
    if (j.contains("tools")) {
        const auto& t = j["tools"];
        if (t.contains("enabled"))        cfg.tools.enabled        = t["enabled"].get<bool>();
        if (t.contains("max_iterations")) cfg.tools.max_iterations = t["max_iterations"].get<int>();
        if (t.contains("working_dir"))    cfg.tools.working_dir    = t["working_dir"].get<std::string>();
    }
    return cfg;
}
