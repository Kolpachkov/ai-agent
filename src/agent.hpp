#pragma once
#include "config.hpp"
#include "tools.hpp"
#include <string>
#include <vector>
#include <functional>

struct Message {
    enum class Role { System, User, Assistant };
    Role        role;
    std::string content;
};

// Return false from callback to abort generation early.
using StreamCallback = std::function<bool(const std::string& piece)>;
// Called each time the model starts generating (initial + after each tool call).
using ThinkCallback  = std::function<void()>;

class Agent {
public:
    explicit Agent(const AppConfig& cfg);
    ~Agent();

    Agent(const Agent&)            = delete;
    Agent& operator=(const Agent&) = delete;

    std::string chat(const std::string& user_message,
                     StreamCallback callback  = nullptr,
                     ThinkCallback  on_think  = nullptr);

    void set_mode(AgentMode mode);   // switch plan/build on the fly
    AgentMode mode() const;

    void reset_history();
    void print_info() const;

    // Persist/restore conversation history (skips system prompt at index 0).
    // Returns number of turns loaded (0 if file missing or disabled).
    int  load_history();
    void save_history() const;

private:
    struct Impl;
    Impl* impl_;

    enum class StopReason { EoT, Eos, ToolCall };

    std::string build_prompt() const;
    std::string active_system_prompt() const;
    void        eval_string(const std::string& text, bool add_bos, bool parse_special);
    std::pair<std::string, StopReason> generate_next(StreamCallback cb);

    const ToolRegistry& active_tools() const;

    std::vector<Message> history_;
    ToolRegistry         tools_;          // build mode: all tools
    ToolRegistry         tools_readonly_; // plan mode: read-only tools
    AppConfig            cfg_;
};
