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

// ── edit_file ─────────────────────────────────────────────────────────────────

static ToolResult impl_edit_file(const std::string& path,
                                  const std::string& old_str,
                                  const std::string& new_str) {
    if (old_str.empty()) return {false, "old_str cannot be empty"};
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) return {false, "cannot open: " + path};
    std::ostringstream ss; ss << fin.rdbuf();
    std::string content = ss.str();
    fin.close();

    const size_t pos = content.find(old_str);
    if (pos == std::string::npos) return {false, "old_str not found in: " + path};
    if (content.find(old_str, pos + 1) != std::string::npos)
        return {false, "old_str found multiple times — add more context to make it unique"};

    content.replace(pos, old_str.size(), new_str);
    std::ofstream fout(path, std::ios::binary);
    if (!fout.is_open()) return {false, "cannot write: " + path};
    fout << content;
    return {true, "edited " + path};
}

// ── python_eval ───────────────────────────────────────────────────────────────

static ToolResult impl_python_eval(const std::string& code, const std::string& working_dir) {
    const std::string tmp = "/tmp/ai_agent_eval_" + std::to_string(getpid()) + ".py";
    { std::ofstream f(tmp); if (!f) return {false, "cannot write temp script"}; f << code; }
    auto r = impl_run_command("python3 " + shell_single_quote(tmp), working_dir);
    std::remove(tmp.c_str());
    return r;
}

// ── memory ────────────────────────────────────────────────────────────────────

static std::string memory_path() {
    const char* h = getenv("HOME");
    return std::string(h ? h : "~") + "/.cache/ai-agent/memory.json";
}

static ToolResult impl_memory_set(const std::string& key, const std::string& value) {
    if (key.empty()) return {false, "key cannot be empty"};
    const std::string path = memory_path();
    nlohmann::json j = nlohmann::json::object();
    { std::ifstream f(path); if (f.is_open()) { try { j = nlohmann::json::parse(f); } catch(...){} } }
    j[key] = value;
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path);
    if (!f) return {false, "cannot write memory file"};
    f << j.dump(2);
    return {true, "memory[" + key + "] set"};
}

static ToolResult impl_memory_get(const std::string& key) {
    const std::string path = memory_path();
    std::ifstream f(path);
    if (!f.is_open()) return {true, key.empty() ? "{}" : "(not set)"};
    nlohmann::json j;
    try { j = nlohmann::json::parse(f); } catch(...) { return {false, "memory file corrupt"}; }
    if (key.empty()) return {true, j.dump(2)};
    if (!j.contains(key)) return {true, "(not set)"};
    return {true, j[key].get<std::string>()};
}

// ── diff_apply ────────────────────────────────────────────────────────────────

static ToolResult impl_diff_apply(const std::string& diff, const std::string& working_dir) {
    if (diff.empty()) return {false, "diff is empty"};
    const std::string tmp = "/tmp/ai_agent_" + std::to_string(getpid()) + ".patch";
    { std::ofstream f(tmp); if (!f) return {false, "cannot write patch file"}; f << diff; }
    auto r = impl_run_command("patch -p1 < " + shell_single_quote(tmp), working_dir);
    std::remove(tmp.c_str());
    return r;
}

// ── tree ──────────────────────────────────────────────────────────────────────

static ToolResult impl_tree(const std::string& path, int max_depth, const std::string& working_dir) {
    const std::string abs = resolve_path(path, working_dir);
    const std::string depth_arg = std::to_string(std::max(1, std::min(max_depth, 10)));
    const std::string cmd =
        "find " + shell_single_quote(abs) + " -maxdepth " + depth_arg +
        " | sort | sed 's|" + abs + "||' | sed '/^$/d' | head -300";
    return impl_run_command(cmd, working_dir);
}

// ── http_request ──────────────────────────────────────────────────────────────

static ToolResult impl_http_request(const std::string& method, const std::string& url,
                                     const std::string& body, const std::string& content_type) {
    if (url.empty()) return {false, "url is required"};
    std::string cmd = "curl -sL --max-time 20 -X " + shell_single_quote(method)
                    + " -H " + shell_single_quote("User-Agent: Mozilla/5.0");
    if (!body.empty()) {
        const std::string ct = content_type.empty() ? "application/json" : content_type;
        cmd += " -H " + shell_single_quote("Content-Type: " + ct)
             + " -d " + shell_single_quote(body);
    }
    cmd += " " + shell_single_quote(url);
    return impl_run_command(cmd, ".");
}

// ── notify ────────────────────────────────────────────────────────────────────

static ToolResult impl_notify(const std::string& title, const std::string& body) {
    return impl_run_command(
        "notify-send " + shell_single_quote(title) + " " + shell_single_quote(body), ".");
}

// ── ask_user ──────────────────────────────────────────────────────────────────

static std::function<std::string(const std::string&)> g_ask_user_fn;
void set_ask_user_fn(std::function<std::string(const std::string&)> fn) { g_ask_user_fn = fn; }

static ToolResult impl_ask_user(const std::string& question) {
    if (!g_ask_user_fn) return {false, "ask_user not available"};
    const std::string answer = g_ask_user_fn(question);
    return {true, answer.empty() ? "(no answer)" : answer};
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
       << "  <tool_call>{\"name\": \"edit_file\", \"arguments\": {\"path\": \"./src/main.cpp\", \"old_str\": \"int x = 1;\", \"new_str\": \"int x = 2;\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"write_file\", \"arguments\": {\"path\": \"./out.txt\", \"content\": \"line1\\nline2\\n\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"run_command\", \"arguments\": {\"command\": \"gcc --version\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"python_eval\", \"arguments\": {\"code\": \"print(2**32)\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"search_files\", \"arguments\": {\"pattern\": \"TODO\", \"directory\": \"./src\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"tree\", \"arguments\": {\"path\": \".\", \"max_depth\": 3}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"web_search\", \"arguments\": {\"query\": \"cmake find_package tutorial\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"fetch_url\", \"arguments\": {\"url\": \"https://example.com\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"http_request\", \"arguments\": {\"method\": \"POST\", \"url\": \"https://api.example.com/data\", \"body\": \"{\\\"key\\\": \\\"value\\\"}\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"memory_set\", \"arguments\": {\"key\": \"project_notes\", \"value\": \"...\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"memory_get\", \"arguments\": {\"key\": \"project_notes\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"ask_user\", \"arguments\": {\"question\": \"Какой файл редактировать?\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"notify\", \"arguments\": {\"title\": \"Готово\", \"body\": \"Сборка завершена\"}}</tool_call>\n"
       << "  <tool_call>{\"name\": \"diff_apply\", \"arguments\": {\"diff\": \"--- a/file.c\\n+++ b/file.c\\n@@ ... \"}}</tool_call>\n\n"
       << "NOTE: In write_file/edit_file, newlines in strings must be \\n (JSON escaped).\n"
       << "PREFER edit_file over write_file for changing existing files — it's safer and uses less context.\n\n"
       << "AUTONOMOUS LOOP:\n"
       << "To continue automatically after finishing a step, end your response with:\n"
       << "  <next>description of the next step</next>\n"
       << "The system will send it back as your next input automatically (user can interrupt with ESC).\n"
       << "Omit <next> when the task is fully complete.\n\n"
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

    reg.add({"edit_file",
             "Replace an exact string in a file (fails if old_str not found or not unique)",
             "{\"path\": \"string\", \"old_str\": \"string\", \"new_str\": \"string\"}",
             [working_dir](const nlohmann::json& args) {
                 return impl_edit_file(resolve_path(args.value("path", ""), working_dir),
                                       args.value("old_str", ""), args.value("new_str", ""));
             }});

    reg.add({"python_eval",
             "Execute Python 3 code and return stdout/stderr",
             "{\"code\": \"string\"}",
             [working_dir](const nlohmann::json& args) {
                 return impl_python_eval(args.value("code", ""), working_dir);
             }});

    reg.add({"memory_set",
             "Persist a key-value pair across sessions (~/.cache/ai-agent/memory.json)",
             "{\"key\": \"string\", \"value\": \"string\"}",
             [](const nlohmann::json& args) {
                 return impl_memory_set(args.value("key", ""), args.value("value", ""));
             }});

    reg.add({"memory_get",
             "Read a persisted value (omit key to dump all memory)",
             "{\"key\": \"string\"}  (key optional)",
             [](const nlohmann::json& args) {
                 return impl_memory_get(args.value("key", ""));
             }});

    reg.add({"diff_apply",
             "Apply a unified diff (patch -p1) in the working directory",
             "{\"diff\": \"string\"}",
             [working_dir](const nlohmann::json& args) {
                 return impl_diff_apply(args.value("diff", ""), working_dir);
             }});

    reg.add({"tree",
             "Show directory tree up to max_depth levels (default 3)",
             "{\"path\": \"string\", \"max_depth\": number}",
             [working_dir](const nlohmann::json& args) {
                 return impl_tree(args.value("path", "."),
                                  args.value("max_depth", 3), working_dir);
             }});

    reg.add({"http_request",
             "Make an HTTP request (GET/POST/PUT/DELETE) and return the response body",
             "{\"method\": \"GET|POST|PUT|DELETE\", \"url\": \"string\", \"body\": \"string\", \"content_type\": \"string\"}",
             [](const nlohmann::json& args) {
                 return impl_http_request(args.value("method", "GET"), args.value("url", ""),
                                          args.value("body", ""), args.value("content_type", ""));
             }});

    reg.add({"notify",
             "Send a desktop notification (notify-send)",
             "{\"title\": \"string\", \"body\": \"string\"}",
             [](const nlohmann::json& args) {
                 return impl_notify(args.value("title", "ai-agent"), args.value("body", ""));
             }});

    reg.add({"ask_user",
             "Pause and ask the user a question; returns their typed answer",
             "{\"question\": \"string\"}",
             [](const nlohmann::json& args) {
                 return impl_ask_user(args.value("question", ""));
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
