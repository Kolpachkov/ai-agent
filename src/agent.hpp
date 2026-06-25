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
    void set_loop_enabled(bool enabled);

    void reset_history();
    void print_info() const;

    // Session persistence.
    // load_session: load from an explicit file path; returns messages loaded.
    // set_session_path: called by main to tell agent where to save (before first turn).
    // save_history: write to current session path (no-op if path is empty).
    int         load_session(const std::string& path);
    void        set_session_path(const std::string& path);
    std::string current_session_path() const;
    void        save_history() const;

private:
    struct Impl;
    Impl* impl_;

    enum class StopReason { EoT, Eos, ToolCall, Aborted };

    std::string session_path_;

    std::string build_prompt() const;
    std::string active_system_prompt() const;
    void        eval_string(const std::string& text, bool add_bos, bool parse_special);
    std::pair<std::string, StopReason> generate_next(StreamCallback cb,
                                                       bool has_tool_turns = false);

    const ToolRegistry& active_tools() const;

    std::vector<Message> history_;
    ToolRegistry         tools_;          // build mode: all tools
    ToolRegistry         tools_readonly_; // plan mode: read-only tools
    AppConfig            cfg_;
    bool                 loop_enabled_ = false;
};
