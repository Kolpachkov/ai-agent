// Integration tests for tool registry (no llama.cpp dependency).
#include "../src/tools.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

static int passed = 0;
static int failed = 0;

#define EXPECT_TRUE(expr) do { \
    if (expr) { ++passed; } else { \
        ++failed; \
        std::cerr << "FAIL [" << __LINE__ << "]: " #expr "\n"; \
    } \
} while(0)

#define EXPECT_EQ(a, b) do { \
    if ((a) == (b)) { ++passed; } else { \
        ++failed; \
        std::cerr << "FAIL [" << __LINE__ << "]: " #a " == " #b "\n" \
                  << "  got:      \"" << (a) << "\"\n" \
                  << "  expected: \"" << (b) << "\"\n"; \
    } \
} while(0)

#define EXPECT_NE(a, b) do { \
    if ((a) != (b)) { ++passed; } else { \
        ++failed; \
        std::cerr << "FAIL [" << __LINE__ << "]: " #a " != " #b "\n"; \
    } \
} while(0)

// Temporary directory RAII
struct TmpDir {
    fs::path path;
    TmpDir() {
        path = fs::temp_directory_path() / ("ai-agent-test-" + std::to_string(getpid()));
        fs::create_directories(path);
    }
    ~TmpDir() { std::error_code ec; fs::remove_all(path, ec); }
};

// ── list_dir tests ────────────────────────────────────────────────────────────

static void test_list_dir_basic(ToolRegistry& reg, const TmpDir& tmp) {
    // Create some files
    std::ofstream(tmp.path / "a.txt") << "hello";
    fs::create_directory(tmp.path / "subdir");

    nlohmann::json args = {{"path", "."}};
    auto r = reg.call("list_dir", args);

    EXPECT_TRUE(r.success);
    EXPECT_NE(r.output.find("a.txt"), std::string::npos);
    EXPECT_NE(r.output.find("subdir"), std::string::npos);
    EXPECT_NE(r.output.find("f "), std::string::npos);  // 'f ' prefix for files
    EXPECT_NE(r.output.find("d "), std::string::npos);  // 'd ' prefix for dirs
}

static void test_list_dir_subpath(ToolRegistry& reg, const TmpDir& tmp) {
    fs::create_directory(tmp.path / "sub");
    std::ofstream(tmp.path / "sub" / "file.cpp") << "code";

    nlohmann::json args = {{"path", "sub"}};
    auto r = reg.call("list_dir", args);

    EXPECT_TRUE(r.success);
    EXPECT_NE(r.output.find("file.cpp"), std::string::npos);
}

static void test_list_dir_nonexistent(ToolRegistry& reg) {
    nlohmann::json args = {{"path", "does_not_exist_xyz"}};
    auto r = reg.call("list_dir", args);
    EXPECT_TRUE(!r.success);
    EXPECT_NE(r.output.find("does not exist"), std::string::npos);
}

static void test_list_dir_empty(ToolRegistry& reg, const TmpDir& tmp) {
    fs::create_directory(tmp.path / "empty_dir");
    nlohmann::json args = {{"path", "empty_dir"}};
    auto r = reg.call("list_dir", args);
    EXPECT_TRUE(r.success);
    EXPECT_NE(r.output.find("(empty)"), std::string::npos);
}

// ── read_file tests ───────────────────────────────────────────────────────────

static void test_read_file_basic(ToolRegistry& reg, const TmpDir& tmp) {
    std::ofstream(tmp.path / "hello.txt") << "world content\n";
    nlohmann::json args = {{"path", "hello.txt"}};
    auto r = reg.call("read_file", args);
    EXPECT_TRUE(r.success);
    EXPECT_NE(r.output.find("world content"), std::string::npos);
}

static void test_read_file_nonexistent(ToolRegistry& reg) {
    nlohmann::json args = {{"path", "no_such_file.txt"}};
    auto r = reg.call("read_file", args);
    EXPECT_TRUE(!r.success);
}

// ── write_file tests ──────────────────────────────────────────────────────────

static void test_write_file_basic(ToolRegistry& reg, const TmpDir& tmp) {
    nlohmann::json args = {{"path", "out.txt"}, {"content", "hello\nworld\n"}};
    auto r = reg.call("write_file", args);
    EXPECT_TRUE(r.success);

    std::ifstream f(tmp.path / "out.txt");
    EXPECT_TRUE(f.is_open());
    std::string content((std::istreambuf_iterator<char>(f)), {});
    EXPECT_EQ(content, std::string("hello\nworld\n"));
}

static void test_write_file_creates_dirs(ToolRegistry& reg, const TmpDir& tmp) {
    nlohmann::json args = {{"path", "deep/nested/file.txt"}, {"content", "data"}};
    auto r = reg.call("write_file", args);
    EXPECT_TRUE(r.success);
    EXPECT_TRUE(fs::exists(tmp.path / "deep" / "nested" / "file.txt"));
}

static void test_write_file_missing_path(ToolRegistry& reg) {
    nlohmann::json args = {{"content", "hello"}};
    auto r = reg.call("write_file", args);
    EXPECT_TRUE(!r.success);
}

// ── run_command tests ─────────────────────────────────────────────────────────

static void test_run_command_basic(ToolRegistry& reg) {
    nlohmann::json args = {{"command", "echo hello_world"}};
    auto r = reg.call("run_command", args);
    EXPECT_TRUE(r.success);
    EXPECT_NE(r.output.find("hello_world"), std::string::npos);
}

static void test_run_command_fail(ToolRegistry& reg) {
    nlohmann::json args = {{"command", "exit 1"}};
    auto r = reg.call("run_command", args);
    EXPECT_TRUE(!r.success);
}

static void test_run_command_pwd(ToolRegistry& reg, const TmpDir& tmp) {
    // run_command should execute in working_dir, not in some arbitrary CWD
    nlohmann::json args = {{"command", "pwd"}};
    auto r = reg.call("run_command", args);
    EXPECT_TRUE(r.success);
    EXPECT_NE(r.output.find(tmp.path.string()), std::string::npos);
}

// ── unknown tool ──────────────────────────────────────────────────────────────

static void test_unknown_tool(ToolRegistry& reg) {
    nlohmann::json args = {};
    auto r = reg.call("rm_rf", args);
    EXPECT_TRUE(!r.success);
    EXPECT_NE(r.output.find("unknown tool"), std::string::npos);
}

// ── readonly registry ─────────────────────────────────────────────────────────

static void test_readonly_no_write(const TmpDir& tmp) {
    auto reg = make_readonly_tools(tmp.path.string());
    EXPECT_TRUE(!reg.has("write_file"));
    EXPECT_TRUE(!reg.has("run_command"));
    EXPECT_TRUE(reg.has("list_dir"));
    EXPECT_TRUE(reg.has("read_file"));
    EXPECT_TRUE(reg.has("search_files"));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    TmpDir tmp;
    auto reg = make_default_tools(tmp.path.string());

    test_list_dir_basic(reg, tmp);
    test_list_dir_subpath(reg, tmp);
    test_list_dir_nonexistent(reg);
    test_list_dir_empty(reg, tmp);

    test_read_file_basic(reg, tmp);
    test_read_file_nonexistent(reg);

    test_write_file_basic(reg, tmp);
    test_write_file_creates_dirs(reg, tmp);
    test_write_file_missing_path(reg);

    test_run_command_basic(reg);
    test_run_command_fail(reg);
    test_run_command_pwd(reg, tmp);

    test_unknown_tool(reg);
    test_readonly_no_write(tmp);

    std::cout << "\nTools tests: " << passed << " passed, " << failed << " failed\n";
    return failed > 0 ? 1 : 0;
}
