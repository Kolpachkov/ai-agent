#pragma once
// Tool-call parsing helpers — no llama.cpp dependency, fully testable.
#include <string>
#include <vector>

static const std::vector<std::string> TOOL_CALL_OPENS = {
    "<tool_call>",   // canonical (what we tell the model to use)
    "<|tool_call>",  // Gemma-style alternate
};
static constexpr const char* TOOL_CALL_OPEN  = "<tool_call>";
static constexpr const char* TOOL_CALL_CLOSE = "</tool_call>";

// Find the earliest occurrence of any tool-call opener in s.
// Sets matched to the string that was found.
inline size_t parsing_find_tool_open(const std::string& s, std::string& matched) {
    size_t best = std::string::npos;
    for (const auto& op : TOOL_CALL_OPENS) {
        const size_t p = s.find(op);
        if (p < best) { best = p; matched = op; }
    }
    return best;
}

// Extract JSON object from a tool call chunk.
// Skips any non-JSON prefix the model may emit (e.g. "call" in <|tool_call>call{...}).
inline std::string parsing_extract_tool_json(const std::string& chunk,
                                              const std::string& matched_open) {
    const size_t a = chunk.find(matched_open);
    if (a == std::string::npos) return {};
    const size_t json_start = chunk.find('{', a + matched_open.size());
    if (json_start == std::string::npos) return {};
    const size_t b = chunk.find(TOOL_CALL_CLOSE, json_start);
    if (b == std::string::npos) return {};
    return chunk.substr(json_start, b - json_start);
}

// Extract text between open and close markers (first occurrence).
inline std::string parsing_extract_between(const std::string& s,
                                            const std::string& open,
                                            const std::string& close) {
    const size_t a = s.find(open);
    if (a == std::string::npos) return {};
    const size_t b = s.find(close, a + open.size());
    if (b == std::string::npos) return {};
    return s.substr(a + open.size(), b - a - open.size());
}
