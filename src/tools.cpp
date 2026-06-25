#include "tools.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/select.h>
#include <ctime>

namespace fs = std::filesystem;

static std::atomic<bool>* g_stop = nullptr;
void set_stop_flag(std::atomic<bool>* f) { g_stop = f; }

// Percent-encode a string for use in URLs.
static std::string url_encode(const std::string& s) {
    std::string r;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
            r += c;
        else { char b[4]; std::snprintf(b, sizeof(b), "%%%02X", c); r += b; }
    }
    return r;
}

// Resolve a path: if relative, anchor it to base_dir.
static std::string resolve_path(const std::string& path, const std::string& base_dir) {
    if (path.empty()) return base_dir;
    const fs::path p(path);
    if (p.is_absolute()) return path;
    return (fs::path(base_dir) / p).lexically_normal().string();
}

// ── Tool implementations (take explicit absolute/resolved paths) ──────────────

static ToolResult impl_read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return {false, "cannot open: " + path};

    std::ostringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();
    if (content.size() > 4000) {
        content.resize(4000);
        content += "\n... [truncated]";
    }
    return {true, content};
}

static ToolResult impl_write_file(const std::string& path, const std::string& content) {
    const fs::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
        if (ec) return {false, "cannot create directories: " + ec.message()};
    }
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return {false, "cannot write: " + path};
    f << content;
    return {true, "wrote " + std::to_string(content.size()) + " bytes → " + path};
}

static std::string shell_single_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else           out += c;
    }
    out += "'";
    return out;
}

static ToolResult impl_run_command(const std::string& cmd, const std::string& working_dir) {
    const std::string safe_cmd =
        "cd " + shell_single_quote(working_dir) +
        " && timeout 30s sh -c " + shell_single_quote(cmd) + " 2>&1";

    int pipefd[2];
    if (pipe(pipefd) < 0) return {false, "pipe failed"};

    const pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]); close(pipefd[1]);
        return {false, "fork failed"};
    }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execl("/bin/sh", "sh", "-c", safe_cmd.c_str(), nullptr);
        _exit(127);
    }

    close(pipefd[1]);
    std::string output;
    char buf[4096];
    bool interrupted = false;

    while (true) {
        fd_set fds; FD_ZERO(&fds); FD_SET(pipefd[0], &fds);
        struct timeval tv = {0, 100000};  // 100ms poll interval
        const int r = select(pipefd[0] + 1, &fds, nullptr, nullptr, &tv);
        if (r < 0) break;
        if (r == 0) {
            if (g_stop && g_stop->load()) { interrupted = true; kill(pid, SIGTERM); break; }
            continue;
        }
        const ssize_t n = read(pipefd[0], buf, sizeof(buf) - 1);
        if (n <= 0) break;
        buf[n] = '\0';
        output += buf;
        if (output.size() > 8000) { output += "\n... [truncated]"; break; }
    }
    close(pipefd[0]);

    if (interrupted) {
        struct timespec ts = {0, 200000000};  // 200ms for graceful exit
        nanosleep(&ts, nullptr);
        int st = 0;
        if (waitpid(pid, &st, WNOHANG) == 0) { kill(pid, SIGKILL); waitpid(pid, &st, 0); }
        return {false, output.empty() ? "[interrupted]" : output + "\n[interrupted]"};
    }

    int status = 0;
    waitpid(pid, &status, 0);
    const int rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (rc != 0) output = "[exit " + std::to_string(rc) + "]\n" + output;
    return {rc == 0, output.empty() ? "(no output)" : output};
}

static ToolResult impl_list_dir(const std::string& path) {
    std::error_code ec;
    if (!fs::exists(path, ec) || ec) return {false, "does not exist: " + path};
    if (!fs::is_directory(path, ec) || ec) return {false, "not a directory: " + path};

    std::vector<std::string> entries;
    std::error_code iter_ec;
    for (const auto& e : fs::directory_iterator(path, iter_ec)) {
        if (iter_ec) {
            return {false, "iteration error: " + iter_ec.message() + " in " + path};
        }
        std::string line = (e.is_directory() ? "d " : "f ");
        line += e.path().filename().string();
        if (!e.is_directory()) {
            std::error_code sz_ec;
            const auto sz = e.file_size(sz_ec);
            if (!sz_ec) line += "  (" + std::to_string(sz) + " B)";
        }
        entries.push_back(std::move(line));
    }
    if (iter_ec) return {false, "iteration error: " + iter_ec.message()};

    std::sort(entries.begin(), entries.end());
    if (entries.empty()) return {true, "(empty)"};
    std::string out;
    for (const auto& s : entries) out += s + "\n";
    return {true, out};
}

static ToolResult impl_search_files(const std::string& pattern,
                                     const std::string& directory,
                                     const std::string& working_dir) {
    const std::string dir = resolve_path(directory, working_dir);
    const std::string cmd =
        "grep -rn "
        "--include='*.cpp' --include='*.hpp' --include='*.h' "
        "--include='*.c'   --include='*.py'  --include='*.js' "
        "--include='*.ts'  --include='*.rs'  --include='*.go' "
        "--include='*.qml' --include='*.cmake' --include='*.md' "
        + shell_single_quote(pattern) + " " + shell_single_quote(dir)
        + " 2>&1 | head -100";
    return impl_run_command(cmd, working_dir);
}

static ToolResult impl_web_search(const std::string& query) {
    if (query.empty()) return {false, "missing: query"};

    static const std::string VENV_PY  = std::string(getenv("HOME") ? getenv("HOME") : "~")
                                        + "/.cache/ai-agent/venv/bin/python3";
    static const std::string SCRIPT   = std::string(getenv("HOME") ? getenv("HOME") : "~")
                                        + "/.cache/ai-agent/search.py";

    auto r = impl_run_command(
        VENV_PY + " " + shell_single_quote(SCRIPT)
        + " " + shell_single_quote(query), ".");

    try {
        auto j = nlohmann::json::parse(r.output);
        if (j.is_object() && j.contains("error"))
            return {false, "search error: " + j["error"].get<std::string>()};
        if (!j.is_array() || j.empty())
            return {true, "No results found"};

        std::string out;
        for (const auto& item : j) {
            const std::string title = item.value("title", "?");
            const std::string url   = item.value("url", "");
            std::string body        = item.value("body", "");
            if (body.size() > 300) body = body.substr(0, 300) + "…";
            out += "[" + title + "](" + url + ")\n";
            if (!body.empty()) out += body + "\n";
            out += "\n";
        }
        return {true, out};
    } catch (...) {
        return {false, "parse error: " + r.output.substr(0, 200)};
    }
}

static ToolResult impl_fetch_url(const std::string& url) {
    if (url.empty()) return {false, "missing: url"};
    const std::string cmd =
        "curl -sL --max-time 15 "
        "-H " + shell_single_quote("User-Agent: Mozilla/5.0 (X11; Linux x86_64) Chrome/120") +
        " " + shell_single_quote(url) +
        " | sed 's/<[^>]*>//g' | sed '/^[[:space:]]*$/d' | head -c 5000";
    return impl_run_command(cmd, ".");
}

// ── Registry ──────────────────────────────────────────────────────────────────

void ToolRegistry::add(Tool tool) { tools_.push_back(std::move(tool)); }
bool ToolRegistry::has(const std::string& name) const {
    for (const auto& t : tools_) if (t.name == name) return true;
    return false;
}
ToolResult ToolRegistry::call(const std::string& name, const nlohmann::json& args) const {
    for (const auto& t : tools_)
        if (t.name == name) return t.fn(args);
    return {false, "unknown tool: " + name};
}
void ToolRegistry::set_working_dir(const std::string& dir) { working_dir_ = dir; }

std::string ToolRegistry::system_section() const {
    std::ostringstream ss;
    ss << "\n\n--- TOOLS ---\n"
       << "Working directory: " << working_dir_ << "\n"
       << "Relative paths resolve from this directory.\n\n"
       << "CALL FORMAT — output exactly, ONE call per turn, then stop:\n"
       << "<tool_call>{\"name\": \"TOOL\", \"arguments\": {\"param\": \"value\"}}</tool_call>\n\n"
       << "EXAMPLES:\n"
       << "  <tool_call>{\"name\": \"list_dir\", \"arguments\": {\"path\": \".\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"./src/main.cpp\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"run_command\", \"arguments\": {\"command\": \"gcc --version\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"search_files\", \"arguments\": {\"pattern\": \"TODO\", \"directory\": \"./src\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"web_search\", \"arguments\": {\"query\": \"cmake find_package tutorial\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"fetch_url\", \"arguments\": {\"url\": \"https://example.com\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"write_file\", \"arguments\": {\"path\": \"./out.txt\", \"content\": \"line1\\nline2\\n\"}}</tool_call>\n\n"
       << "NOTE: In write_file \"content\", newlines must be written as \\n (JSON escaped).\n\n"
       << "AVAILABLE TOOLS:\n";
    for (const auto& t : tools_)
        ss << "  " << t.name << " — " << t.description
           << "\n    args: " << t.params_doc << "\n";
    return ss.str();
}

// ── Factories — each tool captures working_dir and resolves paths itself ──────

ToolRegistry make_default_tools(const std::string& working_dir) {
    ToolRegistry reg;
    reg.set_working_dir(working_dir);

    reg.add({"read_file",
             "Read the contents of a file",
             "{\"path\": \"string\"}",
             [working_dir](const nlohmann::json& args) {
                 return impl_read_file(resolve_path(args.value("path", ""), working_dir));
             }});

    reg.add({"write_file",
             "Write (or overwrite) a file; creates parent directories automatically",
             "{\"path\": \"string\", \"content\": \"string\"}",
             [working_dir](const nlohmann::json& args) {
                 std::string path = args.value("path", "");
                 if (path.empty()) path = args.value("file", "");
                 if (path.empty()) path = args.value("filename", "");
                 if (path.empty()) return ToolResult{false, "missing: path"};
                 return impl_write_file(resolve_path(path, working_dir),
                                        args.value("content", ""));
             }});

    reg.add({"run_command",
             "Run a shell command in the working directory; returns stdout+stderr",
             "{\"command\": \"string\"}",
             [working_dir](const nlohmann::json& args) {
                 const std::string cmd = args.value("command", "");
                 if (cmd.empty()) return ToolResult{false, "missing: command"};
                 return impl_run_command(cmd, working_dir);
             }});

    reg.add({"list_dir",
             "List files and directories at a path",
             "{\"path\": \"string\"}  (default: \".\")",
             [working_dir](const nlohmann::json& args) {
                 return impl_list_dir(resolve_path(args.value("path", "."), working_dir));
             }});

    reg.add({"search_files",
             "Search for a text pattern in source files (grep -rn)",
             "{\"pattern\": \"string\", \"directory\": \"string\"}  (default dir: \".\")",
             [working_dir](const nlohmann::json& args) {
                 return impl_search_files(args.value("pattern", ""),
                                          args.value("directory", "."),
                                          working_dir);
             }});

    reg.add({"web_search",
             "Search the web via SearXNG; returns titles, URLs, and snippets",
             "{\"query\": \"string\"}",
             [](const nlohmann::json& args) {
                 return impl_web_search(args.is_object() ? args.value("query", "") : "");
             }});

    reg.add({"fetch_url",
             "Fetch a web page and return its text content (HTML stripped)",
             "{\"url\": \"string\"}",
             [](const nlohmann::json& args) {
                 return impl_fetch_url(args.is_object() ? args.value("url", "") : "");
             }});

    return reg;
}

ToolRegistry make_readonly_tools(const std::string& working_dir) {
    ToolRegistry reg;
    reg.set_working_dir(working_dir);

    reg.add({"list_dir",
             "List files and directories at a path",
             "{\"path\": \"string\"}  (default: \".\")",
             [working_dir](const nlohmann::json& args) {
                 return impl_list_dir(resolve_path(args.value("path", "."), working_dir));
             }});

    reg.add({"read_file",
             "Read the contents of a file",
             "{\"path\": \"string\"}",
             [working_dir](const nlohmann::json& args) {
                 return impl_read_file(resolve_path(args.value("path", ""), working_dir));
             }});

    reg.add({"search_files",
             "Search for a text pattern in source files (grep -rn)",
             "{\"pattern\": \"string\", \"directory\": \"string\"}  (default dir: \".\")",
             [working_dir](const nlohmann::json& args) {
                 return impl_search_files(args.value("pattern", ""),
                                          args.value("directory", "."),
                                          working_dir);
             }});

    reg.add({"web_search",
             "Search the web via SearXNG; returns titles, URLs, and snippets",
             "{\"query\": \"string\"}",
             [](const nlohmann::json& args) {
                 return impl_web_search(args.is_object() ? args.value("query", "") : "");
             }});

    reg.add({"fetch_url",
             "Fetch a web page and return its text content (HTML stripped)",
             "{\"url\": \"string\"}",
             [](const nlohmann::json& args) {
                 return impl_fetch_url(args.is_object() ? args.value("url", "") : "");
             }});

    return reg;
}
