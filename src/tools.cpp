#include "tools.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <algorithm>

namespace fs = std::filesystem;

// ── Individual tools ──────────────────────────────────────────────────────────

static ToolResult tool_read_file(const nlohmann::json& args) {
    const std::string path = args.value("path", "");
    if (path.empty()) return {false, "missing argument: path"};

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

static ToolResult tool_write_file(const nlohmann::json& args) {
    // Accept "path", "file", or "filename" for robustness
    std::string path = args.value("path", "");
    if (path.empty()) path = args.value("file", "");
    if (path.empty()) path = args.value("filename", "");
    const std::string content = args.value("content", "");
    if (path.empty()) return {false, "missing argument: path — use {\"path\": \"./file.txt\", \"content\": \"...\"}"};

    fs::path p(path);
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

static ToolResult tool_run_command(const nlohmann::json& args) {
    const std::string cmd = args.value("command", "");
    if (cmd.empty()) return {false, "missing argument: command"};

    std::array<char, 4096> buf;
    std::string output;

    // Wrap with timeout to prevent indefinite blocking (30s limit)
    const std::string safe_cmd = "timeout 30s sh -c " + shell_single_quote(cmd) + " 2>&1";
    FILE* pipe = popen(safe_cmd.c_str(), "r");
    if (!pipe) return {false, "popen failed"};

    while (fgets(buf.data(), buf.size(), pipe)) {
        output += buf.data();
        if (output.size() > 4000) {
            output += "\n... [truncated]";
            break;
        }
    }
    const int rc = pclose(pipe);

    if (rc != 0)
        output = "[exit " + std::to_string(rc) + "]\n" + output;

    return {rc == 0, output.empty() ? "(no output)" : output};
}

static ToolResult tool_list_dir(const nlohmann::json& args) {
    const std::string path = args.value("path", ".");

    if (!fs::exists(path))      return {false, "does not exist: " + path};
    if (!fs::is_directory(path)) return {false, "not a directory: " + path};

    std::vector<std::string> entries;
    std::error_code ec2;
    for (const auto& e : fs::directory_iterator(path, ec2)) {
        if (ec2) break;
        std::string line = (e.is_directory() ? "d " : "f ");
        line += e.path().filename().string();
        if (!e.is_directory()) {
            std::error_code sz_ec;
            const auto sz = e.file_size(sz_ec);
            if (!sz_ec) line += "  (" + std::to_string(sz) + " B)";
        }
        entries.push_back(line);
    }
    std::sort(entries.begin(), entries.end());

    if (entries.empty()) return {true, "(empty)"};
    std::string out;
    for (const auto& s : entries) out += s + "\n";
    return {true, out};
}

static ToolResult tool_search_files(const nlohmann::json& args) {
    const std::string pattern   = args.value("pattern", "");
    const std::string directory = args.value("directory", ".");
    if (pattern.empty()) return {false, "missing argument: pattern"};

    const std::string cmd =
        "grep -rn "
        "--include='*.cpp' --include='*.hpp' --include='*.h' "
        "--include='*.c'   --include='*.py'  --include='*.js' "
        "--include='*.ts'  --include='*.rs'  --include='*.go' "
        "--include='*.qml' --include='*.cmake' --include='*.md' "
        + shell_single_quote(pattern) + " " + shell_single_quote(directory)
        + " 2>&1 | head -100";

    return tool_run_command({{"command", cmd}});
}

// ── Registry ──────────────────────────────────────────────────────────────────

void ToolRegistry::add(Tool tool) {
    tools_.push_back(std::move(tool));
}

bool ToolRegistry::has(const std::string& name) const {
    for (const auto& t : tools_) if (t.name == name) return true;
    return false;
}

ToolResult ToolRegistry::call(const std::string& name, const nlohmann::json& args) const {
    for (const auto& t : tools_) {
        if (t.name == name) return t.fn(args);
    }
    return {false, "unknown tool: " + name};
}

void ToolRegistry::set_working_dir(const std::string& dir) {
    working_dir_ = dir;
}

std::string ToolRegistry::system_section() const {
    std::ostringstream ss;
    ss << "\n\n--- TOOLS ---\n"
       << "Working directory: " << working_dir_ << "\n"
       << "IMPORTANT: run_command runs in the working directory above. Use full relative paths.\n"
       << "  Wrong: cd build && cmake ..   (there is no 'build' in working dir)\n"
       << "  Right: cd ./project && cmake -B ./project/build ./project\n"
       << "Call a tool by outputting EXACTLY this format (valid JSON only):\n"
       << "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}</tool_call>\n"
       << "Wait for <tool_response> before making the next call.\n"
       << "CRITICAL for write_file: put file content INSIDE the tool call JSON, never in plain text.\n"
       << "Newlines in content must be escaped as \\n. Example:\n"
       << "<tool_call>{\"name\": \"write_file\", \"arguments\": {\"path\": \"./project/file.txt\", \"content\": \"line1\\nline2\\n\"}}</tool_call>\n\n"
       << "Available tools:\n";
    for (const auto& t : tools_) {
        ss << "  " << t.name << " — " << t.description << "\n"
           << "    args: " << t.params_doc << "\n";
    }
    return ss.str();
}

ToolRegistry make_default_tools(const std::string& working_dir) {
    ToolRegistry reg;

    reg.add({"read_file",
             "Read the contents of a file",
             "{\"path\": \"string\"}",
             tool_read_file});

    reg.add({"write_file",
             "Write (or overwrite) a file; creates parent directories",
             "{\"path\": \"string\", \"content\": \"string\"}",
             tool_write_file});

    reg.add({"run_command",
             "Run a shell command, returns stdout + stderr",
             "{\"command\": \"string\"}",
             tool_run_command});

    reg.add({"list_dir",
             "List files and directories at a path",
             "{\"path\": \"string\"}  (default: \".\")",
             tool_list_dir});

    reg.add({"search_files",
             "Search for a text pattern in source files (grep -rn)",
             "{\"pattern\": \"string\", \"directory\": \"string\"}  (default dir: \".\")",
             tool_search_files});

    reg.set_working_dir(working_dir);
    return reg;
}

ToolRegistry make_readonly_tools(const std::string& working_dir) {
    ToolRegistry reg;

    reg.add({"list_dir",
             "List files and directories at a path",
             "{\"path\": \"string\"}  (default: \".\")",
             tool_list_dir});

    reg.add({"read_file",
             "Read the contents of a file",
             "{\"path\": \"string\"}",
             tool_read_file});

    reg.add({"search_files",
             "Search for a text pattern in source files (grep -rn)",
             "{\"pattern\": \"string\", \"directory\": \"string\"}  (default dir: \".\")",
             tool_search_files});

    reg.set_working_dir(working_dir);
    return reg;
}
