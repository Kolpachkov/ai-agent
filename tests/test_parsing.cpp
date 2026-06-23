// Unit tests for tool-call parsing helpers (no llama.cpp dependency).
#include "../src/parsing.hpp"
#include <cassert>
#include <iostream>
#include <string>

static int passed = 0;
static int failed = 0;

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

// ── parsing_find_tool_open ────────────────────────────────────────────────────

static void test_find_canonical() {
    std::string matched;
    const std::string s = R"(<tool_call>{"name":"list_dir"}</tool_call>)";
    const size_t pos = parsing_find_tool_open(s, matched);
    EXPECT_NE(pos, std::string::npos);
    EXPECT_EQ(matched, std::string("<tool_call>"));
    EXPECT_EQ(pos, size_t(0));
}

static void test_find_alternate() {
    std::string matched;
    const std::string s = R"(<|tool_call>{"name":"list_dir"}</tool_call>)";
    const size_t pos = parsing_find_tool_open(s, matched);
    EXPECT_NE(pos, std::string::npos);
    EXPECT_EQ(matched, std::string("<|tool_call>"));
}

static void test_find_none() {
    std::string matched;
    const std::string s = "some plain text without any tool call";
    const size_t pos = parsing_find_tool_open(s, matched);
    EXPECT_EQ(pos, std::string::npos);
}

static void test_find_earliest() {
    std::string matched;
    // <|tool_call> appears before <tool_call> — should pick the earlier one
    const std::string s = "text <|tool_call>... <tool_call>...";
    const size_t pos = parsing_find_tool_open(s, matched);
    EXPECT_NE(pos, std::string::npos);
    EXPECT_EQ(matched, std::string("<|tool_call>"));
}

static void test_find_with_prefix_text() {
    std::string matched;
    const std::string s = "Вызываю инструмент:\n<tool_call>{\"name\":\"x\"}</tool_call>";
    const size_t pos = parsing_find_tool_open(s, matched);
    EXPECT_NE(pos, std::string::npos);
    EXPECT_EQ(matched, std::string("<tool_call>"));
}

// ── parsing_extract_tool_json ─────────────────────────────────────────────────

static void test_extract_canonical() {
    const std::string chunk = R"(<tool_call>{"name":"list_dir","arguments":{"path":"."}}</tool_call>)";
    const std::string json = parsing_extract_tool_json(chunk, "<tool_call>");
    EXPECT_EQ(json, std::string(R"({"name":"list_dir","arguments":{"path":"."}})"));
}

static void test_extract_with_prefix() {
    // Some models emit "call" or other text between marker and JSON
    const std::string chunk = R"(<|tool_call>call{"name":"run_command","arguments":{"command":"ls"}})</tool_call>)";
    const std::string json = parsing_extract_tool_json(chunk, "<|tool_call>");
    EXPECT_NE(json.find("\"name\""), std::string::npos);
    EXPECT_NE(json.find("\"run_command\""), std::string::npos);
}

static void test_extract_multiline() {
    const std::string chunk =
        "<tool_call>\n"
        "{\n"
        "  \"name\": \"write_file\",\n"
        "  \"arguments\": {\"path\": \"./x.txt\", \"content\": \"hello\\n\"}\n"
        "}\n"
        "</tool_call>";
    const std::string json = parsing_extract_tool_json(chunk, "<tool_call>");
    EXPECT_NE(json.find("write_file"), std::string::npos);
    EXPECT_NE(json.find("hello"), std::string::npos);
}

static void test_extract_missing_close() {
    const std::string chunk = "<tool_call>{\"name\":\"x\"}";
    const std::string json = parsing_extract_tool_json(chunk, "<tool_call>");
    EXPECT_EQ(json, std::string(""));
}

static void test_extract_no_json() {
    const std::string chunk = "<tool_call>no json here</tool_call>";
    const std::string json = parsing_extract_tool_json(chunk, "<tool_call>");
    EXPECT_EQ(json, std::string(""));
}

// ── parsing_extract_between ───────────────────────────────────────────────────

static void test_between_basic() {
    const std::string r = parsing_extract_between("hello <X>world</X> end", "<X>", "</X>");
    EXPECT_EQ(r, std::string("world"));
}

static void test_between_missing_close() {
    const std::string r = parsing_extract_between("hello <X>world", "<X>", "</X>");
    EXPECT_EQ(r, std::string(""));
}

static void test_between_missing_open() {
    const std::string r = parsing_extract_between("hello world</X>", "<X>", "</X>");
    EXPECT_EQ(r, std::string(""));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_find_canonical();
    test_find_alternate();
    test_find_none();
    test_find_earliest();
    test_find_with_prefix_text();

    test_extract_canonical();
    test_extract_with_prefix();
    test_extract_multiline();
    test_extract_missing_close();
    test_extract_no_json();

    test_between_basic();
    test_between_missing_close();
    test_between_missing_open();

    std::cout << "\nParsing tests: " << passed << " passed, " << failed << " failed\n";
    return failed > 0 ? 1 : 0;
}
