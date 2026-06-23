#include "agent.hpp"

#include <llama.h>
#include <nlohmann/json.hpp>

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;

// Chat template: <|turn>role\ncontent<turn|>
// (huihui-ai/Huihui-gemma-4 series)
static constexpr const char* TURN_OPEN  = "<|turn>";
static constexpr const char* TURN_CLOSE = "<turn|>";

// Tool call markers — model may use either form
static const std::vector<std::string> TOOL_OPENS = {
    "<tool_call>",      // canonical form (what we tell the model)
    "<|tool_call>",     // Gemma-style alternate form
};
static constexpr const char* TOOL_OPEN  = "<tool_call>";   // canonical (for injecting)
static constexpr const char* TOOL_CLOSE = "</tool_call>";

// Tool response markers (injected by us, model may echo them — suppress)
static constexpr const char* RESP_OPEN  = "<tool_response>";
static constexpr const char* RESP_CLOSE = "</tool_response>";

// Stray XML blocks the model sometimes emits from training artifacts
// Pattern: <execute_...>...</execute_...> or similar unknown tags
static const std::vector<std::string> STRAY_OPENS  = { "<execute_" };
static const std::vector<std::string> STRAY_CLOSES = { "</execute_" };

// Gemma4 "thinking" channel — suppress internal reasoning
static constexpr const char* THINK_OPEN    = "<|channel>thought";
static constexpr const char* CHANNEL_NEXT  = "<|channel>";  // any following channel ends thinking

// Find earliest occurrence of any tool-call opener; returns pos and which string matched
static size_t find_tool_open(const std::string& s, std::string& matched) {
    size_t best = std::string::npos;
    for (const auto& op : TOOL_OPENS) {
        size_t p = s.find(op);
        if (p < best) { best = p; matched = op; }
    }
    return best;
}

// Extract JSON from tool call (skips any non-JSON prefix like "call")
static std::string extract_tool_json(const std::string& chunk,
                                     const std::string& matched_open) {
    size_t a = chunk.find(matched_open);
    if (a == std::string::npos) return {};
    size_t json_start = chunk.find('{', a + matched_open.size());
    if (json_start == std::string::npos) return {};
    size_t b = chunk.find(TOOL_CLOSE, json_start);
    if (b == std::string::npos) return {};
    return chunk.substr(json_start, b - json_start);
}

// ── Tool call display (Claude Code style) ────────────────────────────────────
static std::string tool_summary(const std::string& name, const ToolResult& r) {
    if (!r.success) {
        std::string msg = r.output.size() > 60 ? r.output.substr(0, 57) + "…" : r.output;
        return "\033[31m✗\033[0m \033[2m" + msg + "\033[0m";
    }
    int lines = (int)std::count(r.output.begin(), r.output.end(), '\n');
    int bytes = (int)r.output.size();
    if (name == "list_dir")
        return "\033[32m✓\033[0m \033[2m" + std::to_string(lines) + " items\033[0m";
    if (name == "read_file")
        return "\033[32m✓\033[0m \033[2m" + (bytes >= 1024
               ? std::to_string(bytes / 1024) + " KB" : std::to_string(bytes) + " B") + "\033[0m";
    if (name == "write_file")
        return "\033[32m✓\033[0m \033[2mwritten\033[0m";
    if (name == "run_command")
        return "\033[32m✓\033[0m \033[2m" + std::to_string(lines) + " lines\033[0m";
    if (name == "search_files")
        return "\033[32m✓\033[0m \033[2m" + std::to_string(lines) + " matches\033[0m";
    return "\033[32m✓\033[0m";
}

static std::string format_tool_line(const std::string& name,
                                    const nlohmann::json& args,
                                    const ToolResult& result) {
    // Build short args string
    std::string arg;
    if (name == "read_file" || name == "write_file")
        arg = args.value("path", "?");
    else if (name == "list_dir")
        arg = args.value("path", ".");
    else if (name == "run_command") {
        arg = args.value("command", "?");
        if (arg.size() > 45) arg = arg.substr(0, 42) + "…";
    } else if (name == "search_files")
        arg = "\"" + args.value("pattern", "?") + "\" in " + args.value("directory", ".");

    return "\n  \033[34m⦿\033[0m \033[1m" + name + "\033[0m"
           "(\033[33m" + arg + "\033[0m)  "
           + tool_summary(name, result) + "\n";
}

// ── Private implementation ────────────────────────────────────────────────────
struct Agent::Impl {
    llama_model*       model   = nullptr;
    llama_context*     ctx     = nullptr;
    llama_sampler*     sampler = nullptr;
    const llama_vocab* vocab   = nullptr;
    int                n_past  = 0;
};

// ── Helpers ───────────────────────────────────────────────────────────────────
static int hw_threads() {
    const int n = static_cast<int>(std::thread::hardware_concurrency());
    return n > 0 ? n : 4;
}

static std::vector<llama_token> tokenize_str(const llama_vocab* vocab,
                                              const std::string& text,
                                              bool add_bos,
                                              bool parse_special) {
    int n_max = static_cast<int>(text.size()) + 128;
    std::vector<llama_token> tokens(n_max);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                           tokens.data(), n_max, add_bos, parse_special);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                           tokens.data(), -n, add_bos, parse_special);
        tokens.resize(n > 0 ? n : 0);
    } else {
        tokens.resize(n);
    }
    return tokens;
}

static std::string extract_between(const std::string& s,
                                    const std::string& open,
                                    const std::string& close) {
    const size_t a = s.find(open);
    if (a == std::string::npos) return {};
    const size_t b = s.find(close, a + open.size());
    if (b == std::string::npos) return {};
    return s.substr(a + open.size(), b - a - open.size());
}

// ── Constructor / Destructor ──────────────────────────────────────────────────
Agent::Agent(const AppConfig& cfg) : cfg_(cfg) {
    impl_ = new Impl();
    // Guard: if anything below throws, free impl_ to avoid a leak.
    // (The destructor does not run when the constructor throws.)
    struct ImplGuard {
        Impl*& ptr;
        bool   armed = true;
        ~ImplGuard() {
            if (!armed) return;
            if (ptr->sampler) llama_sampler_free(ptr->sampler);
            if (ptr->ctx)     llama_free(ptr->ctx);
            if (ptr->model)   llama_model_free(ptr->model);
            delete ptr;
        }
    } guard{impl_};

    if (cfg_.tools.enabled) {
        tools_          = make_default_tools(cfg_.tools.working_dir);
        tools_readonly_ = make_readonly_tools(cfg_.tools.working_dir);
        if (!cfg_.tools.working_dir.empty() && cfg_.tools.working_dir != ".") {
            // Resolve model path to absolute BEFORE chdir so relative paths still work
            cfg_.model.path = fs::absolute(cfg_.model.path).string();
            if (chdir(cfg_.tools.working_dir.c_str()) != 0)
                std::cerr << "[warn] Cannot chdir to working_dir: "
                          << cfg_.tools.working_dir << "\n";
        }
    }

    llama_backend_init();

    // Suppress llama/ggml runtime debug noise (CUDA graph reuse, etc.)
    // Only show warnings and errors; model-load info comes from main.cpp
    const bool verbose = cfg_.agent.verbose;
    llama_log_set([](enum ggml_log_level level, const char* text, void* ud) {
        const bool vb = *static_cast<const bool*>(ud);
        if (vb || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN)
            fputs(text, stderr);
    }, const_cast<bool*>(&cfg_.agent.verbose));
    (void)verbose;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg_.model.n_gpu_layers;
    mparams.use_mmap     = cfg_.model.mmap;
    mparams.use_mlock    = cfg_.model.mlock;

    impl_->model = llama_model_load_from_file(cfg_.model.path.c_str(), mparams);
    if (!impl_->model)
        throw std::runtime_error("Failed to load model: " + cfg_.model.path);
    impl_->vocab = llama_model_get_vocab(impl_->model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = (uint32_t)cfg_.model.n_ctx;
    cparams.n_batch     = (uint32_t)cfg_.model.n_batch;
    cparams.n_threads   = cfg_.model.n_threads < 0 ? hw_threads() : cfg_.model.n_threads;
    cparams.flash_attn_type = cfg_.model.flash_attn
                                  ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                  : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    impl_->ctx = llama_init_from_model(impl_->model, cparams);
    if (!impl_->ctx) {
        llama_model_free(impl_->model);
        throw std::runtime_error("Failed to create llama context");
    }

    impl_->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(impl_->sampler,
        llama_sampler_init_penalties(cfg_.inference.repeat_last_n,
                                     cfg_.inference.repeat_penalty,
                                     0.0f, 0.0f));
    llama_sampler_chain_add(impl_->sampler, llama_sampler_init_top_k(cfg_.inference.top_k));
    llama_sampler_chain_add(impl_->sampler, llama_sampler_init_top_p(cfg_.inference.top_p, 1));
    llama_sampler_chain_add(impl_->sampler, llama_sampler_init_temp(cfg_.inference.temperature));
    const uint32_t seed = cfg_.inference.seed < 0
                              ? (uint32_t)std::time(nullptr)
                              : (uint32_t)cfg_.inference.seed;
    llama_sampler_chain_add(impl_->sampler, llama_sampler_init_dist(seed));

    // Build system prompt based on current mode
    std::string sys = active_system_prompt();
    if (cfg_.tools.enabled)
        sys += active_tools().system_section();
    history_.push_back({Message::Role::System, sys});

    if (cfg_.agent.verbose) print_info();
    guard.armed = false;  // constructor succeeded — disarm the cleanup guard
}

Agent::~Agent() {
    if (impl_->sampler) llama_sampler_free(impl_->sampler);
    if (impl_->ctx)     llama_free(impl_->ctx);
    if (impl_->model)   llama_model_free(impl_->model);
    llama_backend_free();
    delete impl_;
}

// ── Mode helpers ──────────────────────────────────────────────────────────────
std::string Agent::active_system_prompt() const {
    return cfg_.agent.mode == AgentMode::Plan
               ? cfg_.agent.plan_system_prompt
               : cfg_.agent.build_system_prompt;
}

const ToolRegistry& Agent::active_tools() const {
    return cfg_.agent.mode == AgentMode::Plan ? tools_readonly_ : tools_;
}

void Agent::set_mode(AgentMode mode) {
    cfg_.agent.mode = mode;
    // Rebuild system message in history
    std::string sys = active_system_prompt();
    if (cfg_.tools.enabled)
        sys += active_tools().system_section();
    history_[0] = {Message::Role::System, sys};
    // Clear KV cache so new system prompt takes effect
    llama_memory_clear(llama_get_memory(impl_->ctx), true);
    impl_->n_past = 0;
}

AgentMode Agent::mode() const { return cfg_.agent.mode; }

// ── Prompt building ───────────────────────────────────────────────────────────
std::string Agent::build_prompt() const {
    std::string prompt;
    for (const auto& msg : history_) {
        const char* role = nullptr;
        switch (msg.role) {
            case Message::Role::System:    role = "system";    break;
            case Message::Role::User:      role = "user";      break;
            case Message::Role::Assistant: role = "model";     break;
        }
        prompt += std::string(TURN_OPEN) + role + "\n" + msg.content + TURN_CLOSE + "\n";
    }
    prompt += std::string(TURN_OPEN) + "model\n";
    return prompt;
}

// ── KV-cache evaluation ───────────────────────────────────────────────────────
void Agent::eval_string(const std::string& text, bool add_bos, bool parse_special) {
    auto tokens = tokenize_str(impl_->vocab, text, add_bos, parse_special);
    // Guard: n_past + incoming tokens must fit within the context window.
    // Reserve 1024 tokens for model output headroom (analysis can be long).
    constexpr int OUTPUT_HEADROOM = 1024;
    const int capacity = cfg_.model.n_ctx - OUTPUT_HEADROOM;
    if (impl_->n_past + (int)tokens.size() > capacity) {
        throw std::runtime_error(
            "Context overflow: " + std::to_string(impl_->n_past) + " (past) + "
            + std::to_string(tokens.size()) + " (new) tokens exceed n_ctx="
            + std::to_string(cfg_.model.n_ctx) + " — reduce history_limit or "
            "shorten system prompt");
    }
    int n = 0;
    while (n < (int)tokens.size()) {
        const int bs = std::min(cfg_.model.n_batch, (int)tokens.size() - n);
        llama_batch batch = llama_batch_get_one(tokens.data() + n, bs);
        if (llama_decode(impl_->ctx, batch) != 0)
            throw std::runtime_error("llama_decode failed");
        n += bs;
    }
    impl_->n_past += (int)tokens.size();
}

// ── Token generation ──────────────────────────────────────────────────────────
// Streams visible text to `cb`; suppresses <tool_call>, <tool_response> blocks.
// Returns full generated text (with markers) and stop reason.
std::pair<std::string, Agent::StopReason>
Agent::generate_next(StreamCallback cb) {
    // Hold back enough chars to detect any opener without splitting across pieces
    static const int HOLD = 32;

    std::string total;
    std::string pending;
    std::string think_buf;    // accumulates thinking text (for nudge detection)
    bool in_tool     = false;  // inside <tool_call>...</tool_call>
    bool in_response = false;  // inside <tool_response>...</tool_response>
    bool in_think    = false;  // inside <|channel>thought...</|channel>
    bool in_stray    = false;  // inside stray XML block <execute_...>...</execute_...>
    bool think_shown = false;  // emitted the "[думаю...]" header yet
    std::string matched_open;  // which opener was detected


    for (int i = 0; i < cfg_.inference.max_tokens; ++i) {
        llama_token tok = llama_sampler_sample(impl_->sampler, impl_->ctx, -1);
        llama_sampler_accept(impl_->sampler, tok);

        if (llama_vocab_is_eog(impl_->vocab, tok)) {
            if (in_think) {
                think_buf += pending;
            } else if (!in_tool && !in_response && !pending.empty()) {
                if (cb && !cb(pending)) {
                    total += pending;
                    return {total, StopReason::Aborted};
                }
            }
            total += pending;
            return {total, StopReason::Eos};
        }

        char buf[256];
        const int n = llama_token_to_piece(impl_->vocab, tok,
                                            buf, sizeof(buf) - 1, 0, true);
        if (n <= 0) continue;

        pending += std::string(buf, n);

        // ── end-of-turn ───────────────────────────────────────────────────
        const size_t turn_pos = pending.find(TURN_CLOSE);
        if (turn_pos != std::string::npos) {
            const std::string pre = pending.substr(0, turn_pos);
            if (in_think) {
                think_buf += pre;
            } else if (!in_tool && !in_response && !pre.empty()) {
                if (cb && !cb(pre)) {
                    total += pre;
                    return {total, StopReason::Aborted};
                }
            }
            total += pre;
            return {total, StopReason::EoT};
        }

        const bool suppressed = in_tool || in_response || in_think;

        // ── detect thinking block ─────────────────────────────────────────
        if (!suppressed) {
            const size_t tpos = pending.find(THINK_OPEN);
            if (tpos != std::string::npos) {
                const std::string pre = pending.substr(0, tpos);
                if (!pre.empty()) {
                    if (cb && !cb(pre)) {
                        total += pre;
                        return {total, StopReason::Aborted};
                    }
                    total += pre;
                }
                pending = pending.substr(tpos + std::string(THINK_OPEN).size());
                in_think = true;
                think_shown = false;
            }
        }

        // ── suppress thinking, accumulate in think_buf ────────────────────
        if (in_think) {
            if (!think_shown && cb) {
                cb("\033[2m[думаю...]\033[0m\n");  // show indicator only, not content
                think_shown = true;
            }
            // Detect channel switch (end of thinking block)
            const size_t next_ch = pending.find(CHANNEL_NEXT);
            if (next_ch != std::string::npos) {
                think_buf += pending.substr(0, next_ch);
                // Skip past <|channel>xxx\n marker
                const size_t tag_end = pending.find('\n', next_ch);
                const size_t skip_to = tag_end != std::string::npos
                                           ? tag_end + 1
                                           : next_ch + strlen(CHANNEL_NEXT);
                total += pending.substr(0, skip_to);
                pending = pending.substr(skip_to);
                in_think = false;
            } else {
                // Accumulate silently; hold enough bytes to detect <|channel>
                constexpr int THINK_HOLD = 12;
                const int flush_n = (int)pending.size() - THINK_HOLD;
                if (flush_n > 0) {
                    think_buf += pending.substr(0, flush_n);
                    pending = pending.substr(flush_n);
                }
            }
        }

        if (!in_think) {
            // ── detect stray XML blocks (<execute_...>) ───────────────────
            if (!in_tool && !in_response && !in_stray) {
                for (const auto& stray_open : STRAY_OPENS) {
                    const size_t sp = pending.find(stray_open);
                    if (sp != std::string::npos) {
                        const std::string pre = pending.substr(0, sp);
                        if (!pre.empty()) {
                            if (cb && !cb(pre)) {
                                total += pre;
                                return {total, StopReason::Aborted};
                            }
                            total += pre;
                        }
                        pending = pending.substr(sp);
                        in_stray = true;
                        break;
                    }
                }
            }
            if (in_stray) {
                for (const auto& stray_close : STRAY_CLOSES) {
                    const size_t sp = pending.find(stray_close);
                    if (sp != std::string::npos) {
                        // Find end of closing tag (next '>')
                        const size_t tag_end = pending.find('>', sp);
                        total += pending.substr(0, tag_end != std::string::npos ? tag_end + 1 : sp + stray_close.size());
                        pending = pending.substr(tag_end != std::string::npos ? tag_end + 1 : sp + stray_close.size());
                        in_stray = false;
                        break;
                    }
                }
            }

            // ── detect tool call opening ──────────────────────────────────
            if (!in_tool && !in_response && !in_stray) {
                size_t open_pos = find_tool_open(pending, matched_open);
                if (open_pos != std::string::npos) {
                    const std::string pre = pending.substr(0, open_pos);
                    if (!pre.empty()) {
                        if (cb && !cb(pre)) {
                            total += pre;
                            return {total, StopReason::Aborted};
                        }
                        total += pre;
                    }
                    pending = pending.substr(open_pos);
                    in_tool = true;
                }
            }

            // ── detect tool response echoing ──────────────────────────────
            if (!in_tool && !in_response && !in_stray) {
                const size_t ropen_pos = pending.find(RESP_OPEN);
                if (ropen_pos != std::string::npos) {
                    const std::string pre = pending.substr(0, ropen_pos);
                    if (!pre.empty()) {
                        if (cb && !cb(pre)) {
                            total += pre;
                            return {total, StopReason::Aborted};
                        }
                        total += pre;
                    }
                    pending = pending.substr(ropen_pos);
                    in_response = true;
                }
            }

            // ── detect </tool_call> ───────────────────────────────────────
            if (in_tool) {
                const size_t close_pos = pending.find(TOOL_CLOSE);
                if (close_pos != std::string::npos) {
                    total += pending.substr(0, close_pos + std::string(TOOL_CLOSE).size());
                    pending.clear();
                    return {total, StopReason::ToolCall};
                }
            }

            // ── detect </tool_response> ───────────────────────────────────
            if (in_response) {
                const size_t close_pos = pending.find(RESP_CLOSE);
                if (close_pos != std::string::npos) {
                    total += pending.substr(0, close_pos + std::string(RESP_CLOSE).size());
                    pending = pending.substr(close_pos + std::string(RESP_CLOSE).size());
                    in_response = false;
                }
            }
        }

        // ── safe flush visible text ───────────────────────────────────────
        if (!in_tool && !in_response && !in_think && !in_stray) {
            const int flush_n = (int)pending.size() - HOLD;
            if (flush_n > 0) {
                const std::string out = pending.substr(0, flush_n);
                if (cb && !cb(out)) {
                    total += out;
                    pending = pending.substr(flush_n);
                    total += pending;
                    return {total, StopReason::Aborted};
                }
                total += out;
                pending = pending.substr(flush_n);
            }
        }

        llama_batch next = llama_batch_get_one(&tok, 1);
        if (llama_decode(impl_->ctx, next) != 0) {
            // KV cache full — stop cleanly.
            if (in_think) {
                think_buf += pending;
            } else if (!in_tool && !in_response && !pending.empty()) {
                if (cb && !cb(pending))  { total += pending; return {total, StopReason::Aborted}; }
            }
            total += pending;
            break;
        }
        impl_->n_past++;
    }

    // max_tokens reached (or KV break) — flush what we have
    if (in_think) {
        think_buf += pending;
    } else if (!in_tool && !in_response && !pending.empty()) {
        if (cb && !cb(pending)) {
            total += pending;
            return {total, StopReason::Aborted};
        }
    }
    total += pending;
    return {total, StopReason::EoT};
}

// ── Public chat interface ─────────────────────────────────────────────────────
std::string Agent::chat(const std::string& user_msg, StreamCallback cb, ThinkCallback on_think) {
    // Trim old turns if over limit (keep system prompt at index 0) — one O(N) erase
    if ((int)history_.size() > cfg_.agent.history_limit + 1) {
        const int excess = (int)history_.size() - (cfg_.agent.history_limit + 1);
        history_.erase(history_.begin() + 1, history_.begin() + 1 + excess);
    }

    history_.push_back({Message::Role::User, user_msg});

    // Capture only response-channel text for history.
    // Tool status lines (⦿ ...) and thinking bypass capturing_cb entirely.
    // The "\033[2m[думаю...]\033[0m\n" indicator is filtered so it doesn't
    // accumulate in visible_reply.
    static const std::string THINK_IND = "\033[2m[думаю...]\033[0m\n";
    std::string visible_reply;
    StreamCallback capturing_cb = [&](const std::string& piece) -> bool {
        if (piece != THINK_IND)
            visible_reply += piece;
        return cb ? cb(piece) : true;
    };

    std::string full_reply;  // raw content for tool-turn rebuilding only

    const bool tools_active = cfg_.tools.enabled;
    const int  max_iter     = tools_active ? cfg_.tools.max_iterations : 1;

    // Accumulated tool turns for prompt rebuilding each iteration:
    //   .first  = stripped tool call text (no pre-tool narration)
    //   .second = tool result JSON
    std::vector<std::pair<std::string,std::string>> tool_turns;

    // Truncate "output" field in a tool-response JSON to at most max_chars.
    auto trim_resp = [](const std::string& resp_json, int max_chars) -> std::string {
        if (max_chars < 0) return resp_json;
        auto j = nlohmann::json::parse(resp_json, nullptr, false);
        if (j.is_discarded()) return resp_json;
        std::string out = j.value("output", "");
        if ((int)out.size() > max_chars) {
            j["output"]    = out.substr(0, max_chars) + "\n... [урезано]";
            j["truncated"] = true;
        }
        return j.dump(2);
    };

    // Build prompt for the current tool-call iteration with optional nudge turn.
    // max_resp_chars < 0 means no truncation.
    auto build_iter_prompt = [&](int max_resp_chars, bool with_nudge) -> std::string {
        std::string p = build_prompt();
        for (const auto& [gen, resp] : tool_turns) {
            p += gen;
            p += std::string(TURN_CLOSE) + "\n";
            p += std::string(TURN_OPEN) + "user\n";
            p += "<tool_response>\n" + trim_resp(resp, max_resp_chars) + "\n</tool_response>";
            p += std::string(TURN_CLOSE) + "\n";
            p += std::string(TURN_OPEN) + "model\n";
        }
        if (with_nudge) {
            p += std::string(TURN_CLOSE) + "\n";
            p += std::string(TURN_OPEN) + "user\n";
            p += "Напиши подробный анализ на основе полученных данных.";
            p += std::string(TURN_CLOSE) + "\n";
            p += std::string(TURN_OPEN) + "model\n";
        }
        // Always prime into response channel when there are tool turns.
        // Prevents the model from echoing tool_response JSON or opening a
        // thinking block before deciding what to say next.
        if (!tool_turns.empty()) {
            p += std::string(CHANNEL_NEXT) + "response\n";
        }
        return p;
    };

    // Eval prompt with auto-retry on context overflow: progressively truncate
    // tool responses until the prompt fits or all options are exhausted.
    auto eval_with_fallback = [&](bool with_nudge) {
        constexpr int LIMITS[] = {-1, 3000, 1000, 300};
        for (int max_chars : LIMITS) {
            try {
                llama_memory_clear(llama_get_memory(impl_->ctx), true);
                impl_->n_past = 0;
                eval_string(build_iter_prompt(max_chars, with_nudge), true, true);
                return;
            } catch (const std::runtime_error& e) {
                if (std::string(e.what()).find("Context overflow") == std::string::npos)
                    throw;
                // overflow — retry with smaller responses
            }
        }
        throw std::runtime_error(
            "Контекст переполнен даже с минимальными ответами инструментов. "
            "Введите /reset или уменьшите history_limit в config.json.");
    };

    for (int iter = 0; iter < max_iter; ++iter) {
        eval_with_fallback(false);

        if (on_think) on_think();
        auto [chunk, reason] = generate_next(capturing_cb);

        if (reason == StopReason::Aborted) {
            full_reply += chunk;
            break;  // user interrupted — don't nudge, don't retry
        }

        if (reason != StopReason::ToolCall || !tools_active) {
            full_reply += chunk;

            const bool no_response = visible_reply.find_first_not_of(" \t\n\r") == std::string::npos;
            if (no_response && !tool_turns.empty()) {
                // Model produced no visible output after tool calls.
                // Add an explicit nudge turn and prime the response channel
                // so the model generates directly without a thinking block.
                eval_with_fallback(true);

                if (on_think) on_think();
                auto [chunk2, reason2] = generate_next(capturing_cb);
                full_reply += chunk2;
            }
            break;
        }

        // ── Parse tool call ───────────────────────────────────────────────
        std::string matched_op;
        const size_t tool_open_pos = find_tool_open(chunk, matched_op);  // single call, reused below
        const std::string call_json = matched_op.empty()
            ? extract_between(chunk, TOOL_OPEN, TOOL_CLOSE)
            : extract_tool_json(chunk, matched_op);
        if (call_json.empty()) { full_reply += chunk; break; }

        nlohmann::json j = nlohmann::json::parse(call_json, nullptr, false);
        if (j.is_discarded() || !j.contains("name")) { full_reply += chunk; break; }

        const std::string name = j["name"].get<std::string>();
        const nlohmann::json args = j.value("arguments", nlohmann::json::object());

        // ── Execute ───────────────────────────────────────────────────────
        ToolResult result = active_tools().call(name, args);
        if (cb) cb(format_tool_line(name, args, result));  // NOT through capturing_cb

        const std::string resp_json = nlohmann::json{
            {"success", result.success},
            {"output",  result.output}
        }.dump(2);

        full_reply += chunk;

        // Strip pre-tool narration: store only the <tool_call>...</tool_call> part.
        const std::string clean_chunk = (tool_open_pos != std::string::npos)
            ? chunk.substr(tool_open_pos)
            : chunk;
        tool_turns.emplace_back(clean_chunk, resp_json);
    }

    // If the loop exhausted max_iterations with tool calls and never produced a visible
    // reply, force one final generation so the user gets an answer.
    const bool no_reply_yet = visible_reply.find_first_not_of(" \t\n\r") == std::string::npos;
    if (no_reply_yet && !tool_turns.empty()) {
        eval_with_fallback(true);   // rebuild prompt + nudge turn
        if (on_think) on_think();
        auto [final_chunk, _] = generate_next(capturing_cb);
        full_reply += final_chunk;
    }

    // Store only response-channel content in history (no thinking, no tool mechanics).
    // visible_reply contains only non-think, non-tool text from capturing_cb.
    // Fallback to full_reply if visible_reply is whitespace-only.
    const bool has_reply = visible_reply.find_first_not_of(" \t\n\r") != std::string::npos;
    const std::string history_content = has_reply ? visible_reply : full_reply;
    history_.push_back(Message{Message::Role::Assistant, history_content});
    return full_reply;
}

// ── Utility ───────────────────────────────────────────────────────────────────
void Agent::reset_history() {
    std::string sys = active_system_prompt();
    if (cfg_.tools.enabled) sys += active_tools().system_section();

    history_.clear();
    history_.push_back({Message::Role::System, sys});

    llama_memory_clear(llama_get_memory(impl_->ctx), true);
    impl_->n_past = 0;
}

void Agent::print_info() const {
    std::cerr << "[agent] model:      " << cfg_.model.path                    << "\n"
              << "[agent] ctx_train:  " << llama_model_n_ctx_train(impl_->model) << "\n"
              << "[agent] ctx_active: " << llama_n_ctx(impl_->ctx)             << "\n"
              << "[agent] gpu_layers: " << cfg_.model.n_gpu_layers              << "\n"
              << "[agent] tools:      " << (cfg_.tools.enabled ? "on" : "off")  << "\n";
}

// ── Session persistence ───────────────────────────────────────────────────────

static std::string expand_home(const std::string& path) {
    if (path.empty() || path[0] != '~') return path;
    const char* home = getenv("HOME");
    return home ? std::string(home) + path.substr(1) : path;
}

static const char* role_str(Message::Role r) {
    switch (r) {
        case Message::Role::System:    return "system";
        case Message::Role::User:      return "user";
        case Message::Role::Assistant: return "assistant";
    }
    return "user";
}

static Message::Role role_from_str(const std::string& s) {
    if (s == "system")    return Message::Role::System;
    if (s == "assistant") return Message::Role::Assistant;
    return Message::Role::User;
}

int Agent::load_session(const std::string& path) {
    if (path.empty()) return 0;
    std::ifstream f(path);
    if (!f.is_open()) return 0;

    nlohmann::json j;
    try { j = nlohmann::json::parse(f); }
    catch (...) { return 0; }

    if (!j.is_array()) return 0;

    int loaded = 0;
    for (const auto& item : j) {
        if (!item.contains("role") || !item.contains("content")) continue;
        const auto role    = role_from_str(item["role"].get<std::string>());
        const auto content = item["content"].get<std::string>();
        if (role == Message::Role::System) continue;
        history_.push_back({role, content});
        ++loaded;
    }

    if ((int)history_.size() > cfg_.agent.history_limit + 1) {
        const int excess = (int)history_.size() - (cfg_.agent.history_limit + 1);
        history_.erase(history_.begin() + 1, history_.begin() + 1 + excess);
        loaded -= excess;
    }
    return loaded;
}

void Agent::set_session_path(const std::string& path) {
    session_path_ = path;
}

std::string Agent::current_session_path() const {
    return session_path_;
}

void Agent::save_history() const {
    if (session_path_.empty()) return;

    std::error_code ec;
    fs::create_directories(fs::path(session_path_).parent_path(), ec);

    nlohmann::json arr = nlohmann::json::array();
    for (const auto& msg : history_) {
        if (msg.role == Message::Role::System) continue;
        arr.push_back({{"role", role_str(msg.role)}, {"content", msg.content}});
    }

    std::ofstream f(session_path_);
    if (f.is_open()) f << arr.dump(2);
}
